from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.electrical import ElectricalModelState
from ernestogym.ernesto_jax.energy_storage.battery_models.thermal.thermal import ThermalModelState
from ernestogym.ernesto_jax.energy_storage.battery_models.soc import SOCModelState

from flax import struct


@struct.dataclass
class BessState:

    nominal_capacity: float
    nominal_cost: float
    nominal_voltage: float
    nominal_dod: float
    nominal_lifetime: float

    c_max: float
    temp_ambient: float

    v_max: float
    v_min: float

    elapsed_time: float

    electrical_state: ElectricalModelState
    thermal_state: ThermalModelState
    soc_state: SOCModelState
    soh: float