from flax import struct
from functools import partial
import jax

from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.ecm_components.generic_component import GenericComponent, GenericComponentData

@struct.dataclass
class ResistorData(GenericComponentData):
    ...

class Resistor(GenericComponent):

    @classmethod
    def get_initial_state(cls, settings):
        gc_data = super().get_initial_state(settings)
        return ResistorData(**gc_data.__dict__)

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_resistence(cls, res_data:ResistorData, temp: float, soc: float):
        return super().get_value(res_data, temp, soc)