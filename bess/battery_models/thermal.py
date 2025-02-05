from typing import Dict

from flax import struct
from flax.core.frozen_dict import freeze
from functools import partial
import jax

@struct.dataclass
class ThermalModelState:
    c_term: float
    r_cond: float
    r_conv: float
    dv_dT: float

    temp: float
    heat: float
    # soc?

    # v: float
    # i: float
    # v_rc: float

class R2CThermalModel:

    # def __init__(self, components: Dict):
    #     self.r0 = components['r0']
    #     self.rc = freeze(components['rc'])
    #     self.ocv_potential = components['ocv_potential']

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls, components: Dict, temp, heat):
        return ThermalModelState(components['c_term'],
                                 components['r_cond'],
                                 components['r_conv'],
                                 components['dv_dT'],
                                 temp, heat)

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_temp(cls, state: ThermalModelState, q, i, T_amb, dt):

        term_1 = state.c_term / dt * state.temp
        term_2 = T_amb / (state.r_cond + state.r_conv)
        denominator = state.c_term / dt + 1 / (state.r_cond + state.r_conv) - state.dv_dT * i

        t_core = (term_1 + term_2 + q) / denominator

        t_surf = t_core + state.r_cond * (T_amb - t_core) / (state.r_cond + state.r_conv)

        new_state = state.replace(temp=t_surf, heat=q)

        return new_state, t_surf