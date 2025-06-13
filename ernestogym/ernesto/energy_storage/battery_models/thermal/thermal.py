from typing import Dict

import numpy as np
from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

from ernestogym.ernesto.energy_storage.battery_models.lookup_table_maker import build_grid_for_lookup, get_interpolation_1d

@struct.dataclass
class DvdtData:
    lookup_table: jnp.ndarray
    soc_ref: float
    soc_step: float

@struct.dataclass
class ThermalModelState:
    c_term: float
    r_cond: float
    r_conv: float
    dv_dT: DvdtData

    temp: float
    heat: float

class R2CThermalModel:

    @classmethod
    def get_init_state(cls, components_setting: Dict, temp):

        if components_setting['dv_dT']['selected_type'] == 'scalar':
            soc_ref = 0.
            soc_step = 1.
            lookup_table = jnp.array([components_setting['dv_dT']['scalar']])
        elif components_setting['dv_dT']['selected_type'] == 'lookup':
            if 'table' not in components_setting['dv_dT']['lookup'].keys():
                data = np.array([components_setting['dv_dT']['lookup']['inputs']['soc'],
                                 components_setting['dv_dT']['lookup']['outputs']]).T
            else:
                data = components_setting['dv_dT']['lookup']['table']
            lookup_table, soc_ref, soc_step = build_grid_for_lookup(data)
            lookup_table = jnp.asarray(lookup_table)
        else:
            raise ValueError('\'selected_type\' must be \'scalar\' or \'lookup\'')

        dvdt_data = DvdtData(lookup_table=lookup_table, soc_ref=soc_ref, soc_step=soc_step)

        return ThermalModelState(c_term=components_setting['c_term']['scalar'],
                                 r_cond=components_setting['r_cond']['scalar'],
                                 r_conv=components_setting['r_conv']['scalar'],
                                 dv_dT=dvdt_data,
                                 temp=temp,
                                 heat=0.)

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_temp(cls, state: ThermalModelState, q, i, T_amb, soc, dt):

        dv_dT = get_interpolation_1d(state.dv_dT.lookup_table, state.dv_dT.soc_ref, state.dv_dT.soc_step, soc)

        term_1 = state.c_term / dt * state.temp
        term_2 = T_amb / (state.r_cond + state.r_conv)
        denominator = state.c_term / dt + 1 / (state.r_cond + state.r_conv) - dv_dT * i

        t_core = (term_1 + term_2 + q) / denominator

        t_surf = t_core + state.r_cond * (T_amb - t_core) / (state.r_cond + state.r_conv)

        new_state = state.replace(temp=t_surf, heat=q)

        return new_state, t_surf