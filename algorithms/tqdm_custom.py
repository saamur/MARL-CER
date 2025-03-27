import typing

import chex
import jax
import tqdm.auto
import tqdm.notebook
import tqdm.std
from jax.debug import callback


@chex.dataclass
class PBar:
    id: int
    carry: typing.Any


def scan_tqdm(
    pos_carry_in:int,
    pos_carry_out:int,
    pos_index:int,
    n: int,
    print_rate: typing.Optional[int] = None,
    tqdm_type: str = "auto",
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX scan

    Parameters
    ----------
    n : int
        Number of scan steps/iterations.
    print_rate : int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    update_progress_bar, close_tqdm = build_tqdm(n, print_rate, tqdm_type, **kwargs)

    def _scan_tqdm(func):
        """Decorator that adds a tqdm progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `jnp.arange(n)`,
        or be looping over a tuple who's first element is `jnp.arange(n)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(*inputs):

            carry = inputs[pos_carry_in]
            iter_num = inputs[pos_index]

            if isinstance(carry, PBar):
                bar_id = carry.id
                carry = carry.carry
                update_progress_bar(iter_num, bar_id=bar_id)
                result = func(*(inputs[:pos_carry_in] + (carry,) + inputs[pos_carry_in+1:]))
                result = result[:pos_carry_out] + (PBar(id=bar_id, carry=result[pos_carry_out]),) + result[pos_carry_out+1:]
                close_tqdm(iter_num, bar_id=bar_id)
                return result
            else:
                update_progress_bar(iter_num)
                result = func(*inputs)
                close_tqdm(iter_num)
                return result

        return wrapper_progress_bar

    return _scan_tqdm

def build_tqdm(
    n: int,
    print_rate: typing.Optional[int],
    tqdm_type: str,
    **kwargs,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Build the tqdm progress bar on the host

    Parameters
    ----------
    n: int
        Number of updates
    print_rate: int
        Optional integer rate at which the progress bar will be updated,
        If ``None`` the print rate will 1/20th of the total number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    **kwargs
        Extra keyword arguments to pass to tqdm.
    """

    if tqdm_type not in ("auto", "std", "notebook"):
        raise ValueError(
            'tqdm_type should be one of "auto", "std", or "notebook" '
            f'but got "{tqdm_type}"'
        )
    pbar = getattr(tqdm, tqdm_type).tqdm

    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    position_offset = kwargs.pop("position", 0)

    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = dict()

    if print_rate is None:
        if n > 20:
            print_rate = int(n / 20)
        else:
            print_rate = 1
    else:
        if print_rate < 1:
            raise ValueError(f"Print rate should be > 0 got {print_rate}")
        elif print_rate > n:
            raise ValueError(
                "Print rate should be less than the "
                f"number of steps {n}, got {print_rate}"
            )

    remainder = n % print_rate
    remainder = remainder if remainder > 0 else print_rate

    def _define_tqdm(bar_id: int):
        bar_id = int(bar_id)
        tqdm_bars[bar_id] = pbar(
            total=n,
            position=bar_id + position_offset,
            desc=message,
            **kwargs,
        )

    def _update_tqdm(bar_id: int):
        tqdm_bars[int(bar_id)].update(print_rate)

    def _close_tqdm(bar_id: int):
        _pbar = tqdm_bars.pop(int(bar_id))
        _pbar.update(remainder)
        _pbar.clear()
        _pbar.close()

    def update_progress_bar(iter_num: int, bar_id: int = 0):
        """Updates tqdm from a JAX scan or loop"""

        def _inner_init(_i):
            callback(_define_tqdm, bar_id, ordered=True)
            return None

        def _inner_update(i):
            _ = jax.lax.cond(
                i % print_rate == 0,
                lambda: callback(_update_tqdm, bar_id, ordered=True),
                lambda: None,
            )
            return None

        carry = jax.lax.cond(
            iter_num == 0,
            _inner_init,
            _inner_update,
            iter_num,
        )

        return None

    def close_tqdm(iter_num: int, bar_id: int = 0):
        def _inner_close():
            callback(_close_tqdm, bar_id, ordered=True)
            return None

        result = jax.lax.cond(iter_num + 1 == n, _inner_close, lambda : None)
        return None

    return update_progress_bar, close_tqdm
