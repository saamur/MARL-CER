from flax import struct
from functools import partial
import jax

from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.ecm_components.resistor import Resistor, ResistorData
from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.ecm_components.capacitor import Capacitor, CapacitorData

@struct.dataclass
class RCData:
    c: CapacitorData
    r: ResistorData
    r_nominal: ResistorData
    i_resistance: float

class RCParallel:

    @classmethod
    def get_initial_state(cls, settings_r1, settings_c) -> RCData:
        resistor = Resistor.get_initial_state(settings_r1)
        capacitor = Capacitor.get_initial_state(settings_c)

        return RCData(c=capacitor,
                      r=resistor,
                      r_nominal=resistor,
                      i_resistance=0.)

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_resistence(cls, rc_data:RCData, temp: float, soc: float):
        return Resistor.get_resistence(rc_data.r, temp, soc)

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_capacity(cls, rc_data:RCData, temp: float, soc: float):
        return Capacitor.get_capacity(rc_data.c, temp, soc)