from flax import struct
from functools import partial
import jax
import jax.numpy as jnp


@struct.dataclass
class StreamflowState:
    is_init: bool
    cycle_k: int
    last_soc_value: float

    is_direction_up: jnp.ndarray
    is_used: jnp.ndarray
    is_valid: jnp.ndarray
    min_max_vals: jnp.ndarray
    mean_values: jnp.ndarray
    second_signal_means: jnp.ndarray
    number_of_samples: jnp.ndarray
    start_cycles: jnp.ndarray

    iteration: int
    reset_every: int


class Streamflow:

    @classmethod
    @partial(jax.jit, static_argnums=[0, 2])
    def get_init_state(cls, init_soc=1., expected_cycle_num=3000):
        init_state = StreamflowState(is_init=True,
                                     cycle_k=-1,
                                     last_soc_value=init_soc,
                                     is_direction_up=jnp.zeros(expected_cycle_num, dtype=bool),
                                     is_used=jnp.zeros(expected_cycle_num, dtype=bool),
                                     is_valid=jnp.zeros(expected_cycle_num, dtype=bool),
                                     min_max_vals=jnp.zeros((expected_cycle_num, 2)),
                                     mean_values=jnp.zeros(expected_cycle_num),
                                     second_signal_means=jnp.zeros(expected_cycle_num),
                                     number_of_samples=jnp.zeros(expected_cycle_num, dtype=int),
                                     start_cycles=jnp.zeros(expected_cycle_num, dtype=int),
                                     iteration=0,
                                     reset_every=50000)
        return init_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step(cls, state: StreamflowState, actual_soc_value, expected_end, second_signal_value) -> StreamflowState:
        change_direction = jnp.logical_or(state.is_init,
                                          jnp.logical_or(jnp.logical_and(state.is_direction_up[state.cycle_k],
                                                                         actual_soc_value < state.last_soc_value),
                                                         jnp.logical_and(
                                                             jnp.logical_not(state.is_direction_up[state.cycle_k]),
                                                             actual_soc_value > state.last_soc_value)))
        new_is_init = False

        new_state = state.replace(is_init=new_is_init)

        new_state = jax.lax.cond(change_direction,
                                 cls.create_new_cycle,
                                 cls.update_existent_cycle,
                                 new_state, actual_soc_value, second_signal_value, expected_end)

        new_iteration = state.iteration + 1

        new_state = new_state.replace(is_init=new_is_init, last_soc_value=actual_soc_value, iteration=new_iteration)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def create_new_cycle(cls, state: StreamflowState, value: float, second_signal_value: float, expected_end: float):
        new_cycle_k = jnp.sum(state.is_used)

        # FIXME COME FACCIAMO SE VA OLTRE IL LIMITE?

        new_is_valid = state.is_valid.at[new_cycle_k].set(True)
        new_is_used = state.is_used.at[new_cycle_k].set(True)
        new_is_direction_up = state.is_direction_up.at[new_cycle_k].set(value > state.last_soc_value)

        min_val, max_val = jax.lax.min(value, state.last_soc_value), jax.lax.max(value, state.last_soc_value)
        new_min_max_vals = state.min_max_vals.at[new_cycle_k].set((min_val, max_val))

        new_mean_values = state.mean_values.at[new_cycle_k].set(value)
        new_second_signal_means = state.second_signal_means.at[new_cycle_k].set(second_signal_value)

        new_number_of_samples = state.number_of_samples.at[new_cycle_k].set(1)
        new_start_cycles = state.start_cycles.at[new_cycle_k].set(state.iteration)

        new_state = state.replace(is_valid=new_is_valid,
                                  is_used=new_is_used,
                                  is_direction_up=new_is_direction_up,
                                  min_max_vals=new_min_max_vals,
                                  mean_values=new_mean_values,
                                  second_signal_means=new_second_signal_means,
                                  number_of_samples=new_number_of_samples,
                                  start_cycles=new_start_cycles)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def update_existent_cycle(cls, state: StreamflowState, value: float, second_signal_value: float,
                              expected_end: float):
        valid_used_correct_directions = jnp.logical_and(jnp.logical_and(state.is_used, state.is_valid),
                                                        state.is_direction_up == state.is_direction_up[state.cycle_k])

        indices = cls._get_indices_by_direction(direction=state.is_direction_up[state.cycle_k],
                                                min_max_vals=state.min_max_vals,
                                                last_value=state.last_soc_value,
                                                value=value,
                                                indices_range=valid_used_correct_directions)

        expected_end_indices = cls._get_indices_by_direction(direction=state.is_direction_up[state.cycle_k],
                                                             min_max_vals=state.min_max_vals,
                                                             last_value=state.last_soc_value,
                                                             value=expected_end,
                                                             indices_range=valid_used_correct_directions)

        aux_ind = jnp.logical_and(valid_used_correct_directions, indices)

        def when_any_exp_end_ind(state: StreamflowState, aux_ind):
            to_invalid = aux_ind
            return jnp.where(to_invalid, False, state.is_valid), state.cycle_k

        def when_no_exp_end_ind(state: StreamflowState, aux_ind: jnp.ndarray):
            new_is_valid = state.is_valid.at[state.cycle_k].set(False)
            new_cycle_k = jax.lax.cond(state.is_direction_up[state.cycle_k],
                                       lambda mask, min_max_vals: jnp.nanargmax(
                                           jnp.where(mask, min_max_vals[:, 1], jnp.nan)),
                                       lambda mask, min_max_vals: jnp.nanargmin(
                                           jnp.where(mask, min_max_vals[:, 0], jnp.nan)),
                                       aux_ind, state.min_max_vals)
            # indices = jnp.arange(len(new_is_valid))
            # fixme nel codice di Davide è più complesso ma in teoria dovrebbe essere la stessa cosa
            to_invalid = aux_ind
            new_is_valid = jnp.where(to_invalid, False, new_is_valid)
            new_is_valid = new_is_valid.at[new_cycle_k].set(True)
            return new_is_valid, new_cycle_k

        new_is_valid, new_cycle_k = jax.lax.cond(indices.any(),
                                                 lambda state, aux_ind: jax.lax.cond(expected_end_indices.any(),
                                                                                     when_any_exp_end_ind,
                                                                                     when_no_exp_end_ind, state,
                                                                                     aux_ind),
                                                 lambda state, aux_ind: (state.is_valid, state.cycle_k),
                                                 state, aux_ind)
        # new_is_valid, new_cycle_k = jax.lax.cond(expected_end_indices.any(), when_any_exp_end_ind, when_no_exp_end_ind, state, aux_ind)

        new_mean_values = state.mean_values.at[new_cycle_k].set(state.mean_values[new_cycle_k] +
                                                                (value - state.mean_values[new_cycle_k]) /
                                                                (state.number_of_samples[new_cycle_k] + 1))

        new_second_signal_values = state.second_signal_means.at[new_cycle_k].set(
            state.second_signal_means[new_cycle_k] +
            (second_signal_value - state.second_signal_means[new_cycle_k]) /
            (state.number_of_samples[new_cycle_k] + 1))

        new_min_max_vals = state.min_max_vals.at[new_cycle_k].set(
            (jax.lax.min(value, state.min_max_vals[new_cycle_k, 0]),
             jax.lax.max(value, state.min_max_vals[new_cycle_k, 1])))

        new_number_of_samples = state.number_of_samples.at[new_cycle_k].set(state.number_of_samples[new_cycle_k] + 1)
        new_state = state.replace(cycle_k=new_cycle_k,
                                  is_valid=new_is_valid,
                                  min_max_vals=new_min_max_vals,
                                  mean_values=new_mean_values,
                                  second_signal_means=new_second_signal_values,
                                  number_of_samples=new_number_of_samples)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def _get_indices_by_direction(cls, direction: bool, min_max_vals, last_value, value: float,
                                  indices_range: jnp.array):
        indices = jax.lax.cond(direction,
                               lambda min_max_vals, last_value, value: jnp.logical_and(
                                   min_max_vals[:, 1] > last_value,
                                   min_max_vals[:, 1] < value),
                               lambda min_max_vals, last_value, value: jnp.logical_and(
                                   min_max_vals[:, 0] < last_value,
                                   min_max_vals[:, 0] > value),
                               min_max_vals, last_value, value)

        return jnp.logical_and(indices, indices_range)