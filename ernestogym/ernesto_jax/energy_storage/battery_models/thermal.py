from typing import Dict

from flax import struct
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

class R2CThermalModel:

    @classmethod
    # @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls, components_setting: Dict, temp):
        return ThermalModelState(c_term=components_setting['c_term']['scalar'],
                                 r_cond=components_setting['r_cond']['scalar'],
                                 r_conv=components_setting['r_conv']['scalar'],
                                 dv_dT=components_setting['dv_dT']['scalar'],
                                 temp=temp,
                                 heat=0.)

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