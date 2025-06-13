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
    def get_init_state(cls, max_length_reversals) -> DropflowState:
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
                                 lambda state: state.replace(reversals_idx=state.reversals_idx.at[state.reversals_length].set(state.stopper_idx),
                                                             reversals_xs=state.reversals_xs.at[state.reversals_length].set(state.stopper_x),
                                                             reversals_length=state.reversals_length + 1),
                                 state)

        arange = jnp.arange(len(new_state.reversals_xs))

        def body_fun(val):
            i, rev_length, rev_xs, rev_idx, i_start, i_end, cycles, rngs, means, num_cyc = val

            return jax.lax.cond(jnp.abs(rev_xs[i+2] - rev_xs[i+1]) < jnp.abs(rev_xs[i+1] - rev_xs[i]),
                                lambda : (i+1, rev_length, rev_xs, rev_idx, i_start, i_end, cycles, rngs, means, num_cyc),
                                lambda : jax.lax.cond(i == 0,
                                                      lambda: (i, rev_length-1, jnp.where(arange < i, rev_xs, jnp.roll(rev_xs, -1)), jnp.where(arange < i, rev_idx, jnp.roll(rev_idx, -1)),
                                                               i_start.at[num_cyc].set(rev_idx[i]), i_end.at[num_cyc].set(rev_idx[i+1]), cycles.at[num_cyc].set(0.5),
                                                               rngs.at[num_cyc].set(jnp.abs(rev_xs[i]-rev_xs[i+1])), means.at[num_cyc].set(0.5*(rev_xs[i]+rev_xs[i+1])), num_cyc+1),
                                                      lambda: ((i, rev_length-2, jnp.where(arange < i, rev_xs, jnp.roll(rev_xs, -2)), jnp.where(arange < i, rev_idx, jnp.roll(rev_idx, -2)),
                                                               i_start.at[num_cyc].set(rev_idx[i]), i_end.at[num_cyc].set(rev_idx[i+1]), cycles.at[num_cyc].set(1.),
                                                               rngs.at[num_cyc].set(jnp.abs(rev_xs[i]-rev_xs[i+1])), means.at[num_cyc].set(0.5*(rev_xs[i]+rev_xs[i+1])), num_cyc+1))
                                                      )
                                )

        i, rev_length, rev_xs, rev_idx, i_start, i_end, cycles, rngs, means, num_cyc = jax.lax.while_loop(
            lambda val: val[0] < val[1] - 2,
            body_fun,
            (0, new_state.reversals_length, new_state.reversals_xs, new_state.reversals_idx,
             jnp.zeros_like(new_state.reversals_xs, dtype=int), jnp.zeros_like(new_state.reversals_xs, dtype=int),
             jnp.zeros_like(new_state.reversals_xs), jnp.zeros_like(new_state.reversals_xs),
             jnp.zeros_like(new_state.reversals_xs), 0))


        # Else

        i_start = jnp.where(arange < num_cyc, i_start, jnp.roll(rev_idx, num_cyc))
        i_end = jnp.where(arange < num_cyc, i_end, jnp.roll(rev_idx, num_cyc-1))
        rngs = jnp.where(arange < num_cyc, rngs, jnp.abs(jnp.roll(rev_xs, num_cyc) - jnp.roll(rev_xs, num_cyc-1)))
        means = jnp.where(arange < num_cyc, means, 0.5 * (jnp.roll(rev_xs, num_cyc) + jnp.roll(rev_xs, num_cyc - 1)))
        cycles = jnp.where(jnp.logical_and(arange >= num_cyc, arange < (num_cyc + rev_length - 1)),0.5, cycles)

        num_final_cyc = num_cyc
        num_prov_cyc = rev_length - 1

        rev_length = jax.lax.select(jnp.logical_and(rev_xs[rev_length-1] == new_state.stopper_x, rev_idx[rev_length-1] == new_state.stopper_idx),
                                    rev_length-1, rev_length)



        new_state = new_state.replace(reversals_idx=rev_idx,
                                      reversals_xs=rev_xs,
                                      reversals_length=rev_length)

        return new_state, rngs, means, cycles, i_start, i_end, num_final_cyc, num_prov_cyc
