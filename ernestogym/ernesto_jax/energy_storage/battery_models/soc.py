from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

@struct.dataclass
class SOCModelState:
    soc: float
    soc_max: float
    soc_min: float


class SOCModel:

    @classmethod
    # @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls, soc, soc_max, soc_min):
        return SOCModelState(soc=soc, soc_max=soc_max, soc_min=soc_min)

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_soc(cls, state: SOCModelState, i, dt, new_c_max):
        new_soc = state.soc + i / (new_c_max * 3600) * dt
        new_soc = jnp.clip(new_soc, 0, 1)

        new_state = state.replace(soc=new_soc)
        return new_state, new_soc

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def get_feasible_current(cls, state: SOCModelState, soc: float, c_max: float, dt: float):
        """
        Compute the maximum feasible current of the battery according to the soc.
        """
        i_max = (state.soc_max - soc) / dt * c_max * 3600
        i_min = (state.soc_min - soc) / dt * c_max * 3600
        return i_max, i_min