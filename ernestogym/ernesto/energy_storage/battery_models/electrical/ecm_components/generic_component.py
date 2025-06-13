from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

from ernestogym.ernesto.energy_storage.battery_models.lookup_table_maker import build_grid_for_lookup, get_interpolation_2d

@struct.dataclass
class GenericComponentData:
    lookup_table: jnp.ndarray
    temp_ref: float
    soc_ref: float
    temp_step: float
    soc_step: float

class GenericComponent:

    @classmethod
    def get_initial_state(cls, settings):
        if settings['selected_type'] == 'scalar':
            temp_ref = 0.
            soc_ref = 0.
            temp_step = 1.
            soc_step = 1.
            lookup_table = jnp.array([[settings['scalar']]])
        elif settings['selected_type'] == 'lookup':
            lookup_table, temp_ref, soc_ref, temp_step, soc_step = build_grid_for_lookup(settings['lookup']['table'])
            lookup_table = jnp.asarray(lookup_table)
        else:
            raise ValueError('\'selected_type\' must be \'scalar\' or \'lookup\'')

        return GenericComponentData(lookup_table=lookup_table,
                                    temp_ref=temp_ref,
                                    soc_ref=soc_ref,
                                    temp_step=temp_step,
                                    soc_step=soc_step)

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_value(cls, data:GenericComponentData, temp: float, soc: float):
        return get_interpolation_2d(data.lookup_table,
                                    data.temp_ref,
                                    data.soc_ref,
                                    data.temp_step,
                                    data.soc_step,
                                    temp, soc)
