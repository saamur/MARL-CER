from flax import struct
from battery_models.electrical import ElectricalModelState, TheveninModel
from battery_models.thermal import ThermalModelState, R2CThermalModel
from battery_models.soc import SOCModelState, SOCModel
from battery_models.bolun import BolunAgingModel, AgingModelState

from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

@struct.dataclass
class BessState:

    nominal_capacity: float
    c_max: float
    temp_ambient: float

    elapsed_time: float

    electrical_state: ElectricalModelState
    thermal_state: ThermalModelState
    soc_state: SOCModelState
    aging_state: AgingModelState

class BatteryEnergyStorageSystem:

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step(cls, state: BessState, i:float, dt:float):
        new_electrical_state, v_out, _ = TheveninModel.step_current_driven(state.electrical_state, i, dt)

        dissipated_heat = TheveninModel.compute_generated_heat(state.electrical_state)

        new_electrical_state, new_c_max = TheveninModel.compute_parameter_fading(new_electrical_state, state.c_max)

        #todo da aggiornare anche nel soc_model (l'ho fatto in modo che non serva)

        new_thermal_state, curr_temp = R2CThermalModel.compute_temp(state.thermal_state, q=dissipated_heat, i=i, T_amb=state.temp_ambient, dt=dt)
        old_soc = state.soc_state.soc
        new_soc_state, curr_soc = SOCModel.compute_soc(state.soc_state, i, dt, new_c_max)

        new_aging_state, curr_soh = BolunAgingModel.compute_soh(state.aging_state, curr_temp, state.temp_ambient, curr_soc, state.elapsed_time, curr_soc > old_soc)


        new_state = state.replace(c_max=new_c_max,
                                  electrical_state=new_electrical_state,
                                  thermal_state=new_thermal_state,
                                  soc_state=new_soc_state,
                                  aging_state=new_aging_state)

        return new_state