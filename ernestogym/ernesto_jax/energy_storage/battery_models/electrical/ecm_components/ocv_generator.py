from flax import struct
from functools import partial
import jax

from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.ecm_components.generic_component import GenericComponent, GenericComponentData

@struct.dataclass
class OCVData(GenericComponentData):
    ...

class OCVGenerator(GenericComponent):

    @classmethod
    def get_initial_state(cls, settings):
        gc_data = super().get_initial_state(settings)
        return OCVData(**gc_data.__dict__)

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_potential(cls, ocv_data:OCVData, temp: float, soc: float):
        return super().get_value(ocv_data, temp, soc)