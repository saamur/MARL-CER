from typing import Dict

from flax import struct
from flax.core.frozen_dict import freeze
from functools import partial
import jax
import jax.numpy as jnp

@struct.dataclass
class SOCModelState:
    soc: float
    soc_max: float
    soc_min: float


class SOCModel:

    # @classmethod
    # @partial(jax.jit, static_argnums=[0])
    # def get_init_state(cls, components: Dict, temp, heat):
    #     return SOCModel(components['c_term'],
    #                              components['r_cond'],
    #                              components['r_conv'],
    #                              components['dv_dT'],
    #                              temp, heat)

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_soc(cls, state: SOCModelState, i, dt, new_c_max):
        new_soc = state.soc + i / (new_c_max * 3600) * dt
        new_soc = jnp.clip(new_soc, 0, 1)

        new_state = state.replace(soc=new_soc)
        return new_state, new_soc