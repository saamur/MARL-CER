from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

@struct.dataclass
class DropflowState:

    reversals_idx: jnp.ndarray
    reversals_xs: jnp.ndarray
    reversals_length: int

    mean: float
    history_length: int

    idx_last: int
    x_last: float
    x: float
    d_last: float

    stopper_idx: int
    stopper_x: float


class Dropflow:

    @classmethod
    # @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls, max_length_reversals):
        return DropflowState(reversals_idx=jnp.zeros(max_length_reversals, dtype=jnp.int32),
                             reversals_xs=jnp.zeros(max_length_reversals),
                             reversals_length=0,
                             mean=0.,
                             history_length=0,
                             idx_last=0,
                             x_last=0.,
                             x=0.,
                             d_last=0.,
                             stopper_idx=-1,
                             stopper_x=0.)

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def add_point(cls, state: DropflowState, x) -> DropflowState:
        new_state = cls._check_reversal(state, x)

        new_history_length = state.history_length + 1
        new_mean = new_state.mean + (x - new_state.mean) / new_history_length

        new_state = new_state.replace(mean=new_mean, history_length=new_history_length)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def _check_reversal(cls, state: DropflowState, x) -> DropflowState:

        def beginning(state: DropflowState, x):
            new_state = jax.lax.cond(state.history_length == 0,
                                     lambda : state.replace(x_last=x, idx_last=0),
                                     lambda : state.replace(x=x,
                                                            d_last=x-state.x_last,
                                                            reversals_idx=state.reversals_idx.at[state.reversals_length].set(state.idx_last),
                                                            reversals_xs=state.reversals_xs.at[state.reversals_length].set(state.x_last),
                                                            reversals_length=state.reversals_length+1,
                                                            idx_last=1))
            return new_state

        def at_capacity(state: DropflowState, x):

            def main_case(state: DropflowState, x):
                d_next = x - state.x
                new_reversals_idx, new_reversals_xs, new_reversals_length = jax.lax.cond(state.d_last * d_next < 0,
                                                                                         lambda : (state.reversals_idx.at[state.reversals_length].set(state.idx_last),
                                                                                                   state.reversals_xs.at[state.reversals_length].set(state.x),
                                                                                                   state.reversals_length+1),
                                                                                         lambda : (state.reversals_idx,
                                                                                                   state.reversals_xs,
                                                                                                   state.reversals_length))

                return state.replace(reversals_idx=new_reversals_idx,
                                     reversals_xs=new_reversals_xs,
                                     reversals_length=new_reversals_length,
                                     x_last=state.x,
                                     x=x,
                                     d_last=d_next,
                                     idx_last=state.history_length,
                                     stopper_idx=state.history_length,
                                     stopper_x=x)

            new_state = jax.lax.cond(x == state.x,
                                     lambda state, x: state.replace(idx_last=state.history_length),
                                     main_case,
                                     state, x)

            return new_state

        new_state = jax.lax.cond(state.history_length < 2, beginning, at_capacity, state, x)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def extract_new_cycles(cls, state: DropflowState):

        new_state = jax.lax.cond(state.stopper_idx == -1,
                                 lambda state: state,
                                 lambda state: state.replace(reversals_idx=state.reversals_idx.at[state.reversals_length].set(state.idx_last),
                                                             reversals_xs=state.reversals_xs.at[state.reversals_length].set(state.x),
                                                             reversals_length=state.reversals_length + 1),
                                 state)

        #todo check len revs (dovrebbe non servire)

        abs_diffs = jnp.abs(jnp.diff(new_state.reversals_xs))       # n - 1

        diffs_prev = abs_diffs[:-1]                                 # n - 2
        diffs_next = abs_diffs[1:]                                  # n - 2

        condition = diffs_next < diffs_prev

        def body_fun(val):
            valid_revs, i, half, cycles = val

            return jax.lax.cond(condition[i],
                                lambda : (valid_revs, i+1, False, cycles),
                                lambda : jax.lax.cond(half,
                                                      lambda: (valid_revs.at[i].set(False), i+1, True, cycles.at[i].set(0.5)),
                                                      lambda: (valid_revs.at[i].set(False).at[i+1].set(False), i+2, False, cycles.at[i].set(1))
                                                      )
                                )

        valid_revs, i, half, cycles = jax.lax.while_loop(lambda val : val[1] < new_state.reversals_length-2,
                                                         body_fun,
                                                         (jnp.ones_like(new_state.reversals_xs, dtype=bool), 0, True, jnp.zeros_like(new_state.reversals_xs)))

        # Else

        cycles = jnp.where(jnp.logical_and(valid_revs, jnp.arange(len(valid_revs)) < new_state.reversals_length), 0.5, cycles)
        # cycles = cycles.at[new_state.reversals_length-1].set(0)

        new_reversals_length = jax.lax.cond(jnp.logical_and(new_state.reversals_xs[new_state.reversals_length-1] == new_state.stopper_x, new_state.reversals_idx[new_state.reversals_length-1] == new_state.stopper_idx),
                                            lambda : new_state.reversals_length-1,
                                            lambda : new_state.reversals_length)

        rngs = abs_diffs
        means = 0.5 * (new_state.reversals_xs[:-1] + new_state.reversals_xs[1:])
        counts = cycles[:-1]
        i_start = new_state.reversals_idx[:-1]
        i_end = new_state.reversals_idx[1:]


        #compact arrays:
        mask = jnp.logical_and(valid_revs, jnp.arange(len(valid_revs)) < new_reversals_length)

        indexes = jnp.argsort(jnp.logical_not(mask))
        new_reversals_length = mask.sum()

        new_state = new_state.replace(reversals_idx=new_state.reversals_idx[indexes],
                                      reversals_xs=new_state.reversals_xs[indexes],
                                      reversals_length=new_reversals_length)

        return new_state, rngs, means, counts, i_start, i_end