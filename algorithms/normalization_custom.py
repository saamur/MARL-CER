import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Array,
  Dtype,
  Initializer,
  Axes,
)


def _canonicalize_axes(rank: int, axes: Axes) -> tp.Tuple[int, ...]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, tp.Iterable):
    axes = (axes,)
  return tuple({rank + axis if axis < 0 else axis for axis in axes})


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(
  x: Array,
  axes: Axes,
  dtype: tp.Optional[Dtype],
  axis_name: tp.Optional[str] = None,
  axis_index_groups: tp.Any = None,
  use_mean: bool = True,
  use_fast_variance: bool = True,
  mask: tp.Optional[Array] = None,
):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for stability in half precision training.
  - If ``use_fast_variance`` is ``True``, mean and variance are computed using
    Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single
    XLA fusion.
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single ``lax.pmean`` call to avoid latency.

  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    dtype: Optional dtype specifying the minimal precision. Statistics are
      always at least float32 for stability (default: dtype of x).
    axis_name: Optional name for the pmapped axis to compute mean over. Note,
      this is only used for pmap and shard map. For SPMD jit, you do not need to
      manually synchronize. Just make sure that the axes are correctly annotated
      and XLA:SPMD will insert the necessary collectives.
    axis_index_groups: Optional axis indices.
    use_mean: If true, calculate the mean from the input and use it when
      computing the variance. If false, set the mean to zero and compute the
      variance without subtracting the mean.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
    mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
      the positions for which the mean and variance should be computed.

  Returns:
    A pair ``(mean, var)``.
  """
  if dtype is None:
    dtype = jnp.result_type(x)
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  dtype = jnp.promote_types(dtype, jnp.float32)
  x = jnp.asarray(x, dtype)
  axes = _canonicalize_axes(x.ndim, axes)

  def maybe_distributed_mean(*xs, mask=None):
    mus = tuple(x.mean(axes, where=mask) for x in xs)
    if axis_name is None:
      return mus if len(xs) > 1 else mus[0]
    else:
      # In the distributed case we stack multiple arrays to speed comms.
      if len(xs) > 1:
        reduced_mus = lax.pmean(
          jnp.stack(mus, axis=0),
          axis_name,
          axis_index_groups=axis_index_groups,
        )
        return tuple(reduced_mus[i] for i in range(len(xs)))
      else:
        return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

  if use_mean:
    if use_fast_variance:
      mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
      # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
      # to floating point round-off errors.
      var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
    else:
      mu = maybe_distributed_mean(x, mask=mask)
      var = maybe_distributed_mean(
        _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
      )
  else:
    var = maybe_distributed_mean(_abs_sq(x), mask=mask)
    mu = jnp.zeros_like(var)
  return mu, var


def _normalize(
  x: Array,
  mean: Array,
  var: Array,
  scale: tp.Optional[Array],
  bias: tp.Optional[Array],
  reduction_axes: Axes,
  feature_axes: Axes,
  dtype: tp.Optional[Dtype],
  epsilon: float,
):
  """ "Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    epsilon: Normalization epsilon.

  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  args = [x]
  if scale is not None:
    scale = scale.reshape(feature_shape)
    mul *= scale
    args.append(scale)
  y *= mul
  if bias is not None:
    bias = bias.reshape(feature_shape)
    y += bias
    args.append(bias)
  dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


class RunningNorm(Module):
  """BatchNorm Module.

  To calculate the batch norm on the input and update the batch statistics,
  call the :func:`train` method (or pass in ``use_running_average=False`` in
  the constructor or during call time).

  To use the stored batch statistics' running average, call the :func:`eval`
  method (or pass in ``use_running_average=True`` in the constructor or
  during call time).

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (5, 6))
    >>> layer = nnx.BatchNorm(num_features=6, momentum=0.9, epsilon=1e-5,
    ...                       dtype=jnp.float32, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': VariableState(
        type=Param,
        value=(6,)
      ),
      'mean': VariableState(
        type=BatchStat,
        value=(6,)
      ),
      'scale': VariableState(
        type=Param,
        value=(6,)
      ),
      'var': VariableState(
        type=BatchStat,
        value=(6,)
      )
    })

    >>> # calculate batch norm on input and update batch statistics
    >>> layer.train()
    >>> y = layer(x)
    >>> batch_stats1 = nnx.state(layer, nnx.BatchStat)
    >>> y = layer(x)
    >>> batch_stats2 = nnx.state(layer, nnx.BatchStat)
    >>> assert (batch_stats1['mean'].value != batch_stats2['mean'].value).all()
    >>> assert (batch_stats1['var'].value != batch_stats2['var'].value).all()

    >>> # use stored batch statistics' running average
    >>> layer.eval()
    >>> y = layer(x)
    >>> batch_stats3 = nnx.state(layer, nnx.BatchStat)
    >>> assert (batch_stats2['mean'].value == batch_stats3['mean'].value).all()
    >>> assert (batch_stats2['var'].value == batch_stats3['var'].value).all()

  Args:
    num_features: the number of input features.
    use_running_average: if True, the stored batch statistics will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over
      the examples on the first two and last two devices. See ``jax.lax.psum``
      for more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
    rngs: rng key.
  """

  def __init__(
    self,
    num_features: int,
    *,
    use_running_average: bool = False,
    axis: int = -1,
    epsilon: float = 1e-5,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    feature_shape = (num_features,)
    self.mean = nnx.BatchStat(jnp.zeros(feature_shape, dtype=float))
    self.sum_squared_differences = nnx.BatchStat(jnp.ones(feature_shape, dtype=float))
    self.count = nnx.BatchStat(jnp.zeros(shape=(), dtype=int))

    self.scale: nnx.Param[jax.Array] | None
    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = None

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = None

    self.num_features = num_features
    self.use_running_average = use_running_average
    self.axis = axis
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(
    self,
    x,
    use_running_average: tp.Optional[bool] = None,
    *,
    mask: tp.Optional[jax.Array] = None,
  ):
    """Normalizes the input using batch statistics.

    Args:
      x: the input to be normalized.
      use_running_average: if true, the stored batch statistics will be
        used instead of computing the batch statistics on the input. The
        ``use_running_average`` flag passed into the call method will take
        precedence over the ``use_running_average`` flag passed into the
        constructor.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    use_running_average = first_from(
      use_running_average,
      self.use_running_average,
      error_msg="""No `use_running_average` argument was provided to BatchNorm
        as either a __call__ argument, class attribute, or nnx.flag.""",
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

    if not use_running_average:
      self.welford(x)


    return _normalize(
      x,
      self.mean.value,
      self.population_variance(),
      self.scale.value if self.scale else None,
      self.bias.value if self.bias else None,
      reduction_axes,
      feature_axes,
      self.dtype,
      self.epsilon,
    )

  def welford(self, x):

    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    batch_dims = tuple(i for i in range(x.ndim) if i not in feature_axes)
    # num_x = jnp.prod(np.array(x.shape)[batch_dims])
    num_x = np.prod([x.shape[dim] for dim in batch_dims])
    old_count = self.count.value
    self.count.value += num_x

    batch_mean = x.mean(axis=batch_dims)

    self.mean.value += num_x/self.count.value * (batch_mean - self.mean.value)

    batch_sum_squared_differences = jnp.sum(jnp.square(x - jnp.expand_dims(batch_mean, axis=batch_dims)), axis=batch_dims)

    self.sum_squared_differences += batch_sum_squared_differences + (old_count * num_x / self.count.value) * jnp.square(batch_mean - self.mean.value)

  def population_variance(self):
    return self.sum_squared_differences.value / self.count.value

class MaskedRunningNorm(Module):

  def __init__(
    self,
    num_features: int,
    *,
    mask: tp.Sequence[bool] = None,
    use_running_average: bool = False,
    axis: int = -1,
    epsilon: float = 1e-5,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):

    if mask is None:
      mask = [True] * num_features

    self.mask = tuple(mask)

    feature_shape = (num_features,)
    norm_feature_shape = (np.sum(self.mask),)

    self.norm_mean = nnx.BatchStat(jnp.zeros(norm_feature_shape, jnp.float32))
    self.norm_sum_squared_differences = nnx.BatchStat(jnp.ones(norm_feature_shape, jnp.float32))
    self.count = nnx.BatchStat(jnp.zeros(shape=(), dtype=jnp.int64))

    self.scale: nnx.Param[jax.Array] | None
    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = None

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = None


    self.num_features = num_features
    self.use_running_average = use_running_average
    self.axis = axis
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(
    self,
    x,
    use_running_average: tp.Optional[bool] = None,
    *,
    mask: tp.Optional[jax.Array] = None,
  ):

    use_running_average = first_from(
      use_running_average,
      self.use_running_average,
      error_msg="""No `use_running_average` argument was provided to BatchNorm
        as either a __call__ argument, class attribute, or nnx.flag.""",
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

    if not use_running_average:
      self.welford(x)


    return _normalize(
      x,
      self.mean(),
      self.population_variance(),
      self.scale.value if self.scale else None,
      self.bias.value if self.bias else None,
      reduction_axes,
      feature_axes,
      self.dtype,
      self.epsilon,
    )

  def welford(self, x):

    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    # batch_dims = tuple(i for i in range(x.ndim) if i not in feature_axes)

    n_feature_dims = len(feature_axes)

    x_features_first = jnp.moveaxis(x, feature_axes, np.arange(n_feature_dims))
    batch_dims = tuple(i for i in range(n_feature_dims, x.ndim))

    # num_x = jnp.prod(np.array(x.shape)[batch_dims])
    num_x = np.prod([x_features_first.shape[dim] for dim in batch_dims])
    old_count = self.count.value
    self.count.value += num_x

    print(np.array(self.mask).shape)

    # broad_mask = np.expand_dims(np.array(self.mask), axis=batch_dims)
    # broad_mask = np.broadcast_to(broad_mask, x.shape)
    broad_mask = np.array(self.mask)
    print(broad_mask.shape)

    print('bm', x_features_first[broad_mask].shape)

    batch_mean = x_features_first[broad_mask].mean(axis=batch_dims)

    self.norm_mean.value += num_x / self.count.value * (batch_mean - self.norm_mean.value)

    batch_sum_squared_differences = jnp.sum(jnp.square(x_features_first[broad_mask] - jnp.expand_dims(batch_mean, axis=batch_dims)), axis=batch_dims)

    self.norm_sum_squared_differences += batch_sum_squared_differences + (old_count * num_x / self.count.value) * jnp.square(batch_mean - self.norm_mean.value)

  def mean(self):
    return jnp.zeros((self.num_features,)).at[np.array(self.mask)].set(self.norm_mean.value)

  def population_variance(self):
    return jnp.ones((self.num_features,)).at[np.array(self.mask)].set(self.norm_sum_squared_differences.value / self.count.value)