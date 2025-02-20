from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

from ernestogym.ernesto_jax.energy_storage.battery_models.lookup_table_maker import build_grid_for_lookup, get_interpolation_2d

@struct.dataclass
class CapacitorData:
    lookup_table: jnp.ndarray
    temp_step: float
    soc_step: float

class Capacitor:

    @classmethod
    def get_initial_state(cls, settings):
        if settings['selected_type'] == 'scalar':
            temp_step = 1.
            soc_step = 1.
            lookup_table = jnp.array([[settings['scalar']]])
        elif settings['selected_type'] == 'lookup':
            lookup_table, temp_step, soc_step = build_grid_for_lookup(settings['lookup']['table'])
            lookup_table = jnp.asarray(lookup_table)
        else:
            raise ValueError('\'selected_type\' must be \'scalar\' or \'lookup\'')

        return CapacitorData(lookup_table=lookup_table,
                             temp_step=temp_step,
                             soc_step=soc_step)

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_capacity(cls, cap_data:CapacitorData, temp: float, soc: float):
        return get_interpolation_2d(cap_data.lookup_table, cap_data.temp_step, cap_data.soc_step, temp, soc)