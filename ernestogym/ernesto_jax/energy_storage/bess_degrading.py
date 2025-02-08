from ernestogym.ernesto_jax.energy_storage.battery_models.electrical import ElectricalModelState, TheveninModel
from ernestogym.ernesto_jax.energy_storage.battery_models.thermal import ThermalModelState, R2CThermalModel
from ernestogym.ernesto_jax.energy_storage.battery_models.soc import SOCModelState, SOCModel
from ernestogym.ernesto_jax.energy_storage.battery_models.bolun_streamflow import BolunStreamflowModel, BolunStreamflowState

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
    aging_state: BolunStreamflowState

class BatteryEnergyStorageSystem:

    @classmethod
    def get_init_state(cls,
                       models_config: list,
                       battery_options: dict,
                       input_var: str):

        assert input_var == 'current'

        nominal_capacity = battery_options['params']['nominal_capacity']
        c_max = battery_options['params']['nominal_capacity']

        # nominal_dod ?
        # nominal_lifetime ?
        # nominal_voltage ?
        # nominal_cost ?
        # v_max, v_min ?
        temp_ambient = battery_options['params']['temp_ambient']
        # soc_min soc_max
        sign_convention = battery_options['sign_convention']
        # _reset_soc_every ?
        # _check_soh_every ? but not applicable

        soc_min = battery_options['bounds']['soc']['low']
        soc_max = battery_options['bounds']['soc']['high']
        soc = battery_options['init']['soc']

        soc_state = SOCModel.get_init_state(soc=soc, soc_min=soc_min, soc_max=soc_max)

        temp_battery = battery_options['init']['temperature']

        electrical_state, thermal_state, aging_state = cls._build_models(models_config, sign_convention, temp_battery)

        init_state = BessState(nominal_capacity=nominal_capacity,
                               c_max=c_max,
                               temp_ambient=temp_ambient,
                               elapsed_time=0.,
                               electrical_state=electrical_state,
                               thermal_state=thermal_state,
                               soc_state=soc_state,
                               aging_state=aging_state)

        init_state = jax.tree.map(lambda leaf: jnp.array(leaf), init_state)

        return init_state

    @classmethod
    def _build_models(cls, models_settings, sign_convention, temp_battery):
        electrical_state = None
        thermal_state = None
        aging_state = None

        for model_config in models_settings:
            if model_config['type'] == 'electrical':
                assert model_config['class_name'] == 'TheveninModel'
                assert model_config['use_fading'] == False
                electrical_state = TheveninModel.get_init_state(components=model_config['components'],
                                                                sign_convention=sign_convention)

            elif model_config['type'] == 'thermal':
                assert model_config['class_name'] == 'R2CThermal'
                thermal_state = R2CThermalModel.get_init_state(model_config['components'], temp_battery)

            elif model_config['type'] == 'aging':
                assert model_config['class_name'] == 'BolunModel'
                aging_state = BolunStreamflowModel.get_init_state(components_setting=model_config['components'],
                                                                  stress_models=model_config['stress_models'],
                                                                  temp=temp_battery)

        return electrical_state, thermal_state, aging_state


    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step(cls, state: BessState, i:float, dt:float):
        new_electrical_state, v_out, _ = TheveninModel.step_current_driven(state.electrical_state, i, dt)

        dissipated_heat = TheveninModel.compute_generated_heat(state.electrical_state)

        #todo da aggiornare anche nel soc_model (l'ho fatto in modo che non serva)

        new_thermal_state, curr_temp = R2CThermalModel.compute_temp(state.thermal_state, q=dissipated_heat, i=i, T_amb=state.temp_ambient, dt=dt)
        old_soc = state.soc_state.soc
        new_soc_state, curr_soc = SOCModel.compute_soc(state.soc_state, i, dt, state.c_max)

        new_aging_state, curr_soh = BolunStreamflowModel.compute_soh(state.aging_state, curr_temp, state.temp_ambient, curr_soc, state.elapsed_time, curr_soc > old_soc)

        new_c_max = curr_soh * state.nominal_capacity

        new_state = state.replace(electrical_state=new_electrical_state,
                                  thermal_state=new_thermal_state,
                                  soc_state=new_soc_state,
                                  aging_state=new_aging_state,
                                  c_max=new_c_max)

        return new_state