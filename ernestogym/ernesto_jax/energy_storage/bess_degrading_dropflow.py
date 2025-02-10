from ernestogym.ernesto_jax.energy_storage.battery_models.electrical import ElectricalModelState, TheveninModel
from ernestogym.ernesto_jax.energy_storage.battery_models.thermal import ThermalModelState, R2CThermalModel
from ernestogym.ernesto_jax.energy_storage.battery_models.soc import SOCModelState, SOCModel
from ernestogym.ernesto_jax.energy_storage.battery_models.bolun_dropflow import BolunDropflowModel, BolunDropflowState
from ernestogym.ernesto_jax.energy_storage.bess import BessState

from flax import struct
from functools import partial
import jax
import jax.numpy as jnp


@struct.dataclass
class BessBolunDropflowState(BessState):
    aging_state: BolunDropflowState
    iter: int
    check_soh_every: int

class BatteryEnergyStorageSystem:

    @classmethod
    def get_init_state(cls,
                       models_config: list,
                       battery_options: dict,
                       input_var: str,
                       check_soh_every: int=100):

        assert input_var == 'current'

        nominal_capacity = battery_options['params']['nominal_capacity']
        c_max = battery_options['params']['nominal_capacity']
        nominal_cost = battery_options['params']['nominal_cost']
        nominal_voltage = battery_options['params']['nominal_voltage']
        nominal_dod = battery_options['params']['nominal_dod']
        nominal_lifetime = battery_options['params']['nominal_lifetime']

        # nominal_dod ?
        # nominal_lifetime ?
        # nominal_voltage ?
        # nominal_cost ?
        v_max = battery_options['params']['v_max']
        v_min = battery_options['params']['v_min']
        temp_ambient = battery_options['init']['temp_ambient']
        # soc_min soc_max
        sign_convention = battery_options['sign_convention']
        # _reset_soc_every ?
        # _check_soh_every ? but not applicable

        soc_min = battery_options['bounds']['soc']['low']
        soc_max = battery_options['bounds']['soc']['high']
        soc = battery_options['init']['soc']

        soc_state = SOCModel.get_init_state(soc=soc, soc_min=soc_min, soc_max=soc_max)

        temp_battery = battery_options['init']['temperature']

        electrical_state, thermal_state, aging_state = cls._build_models(models_config, battery_options['init'], sign_convention, temp_battery)

        init_state = BessBolunDropflowState(nominal_capacity=nominal_capacity,
                                            nominal_cost=nominal_cost,
                                            nominal_voltage=nominal_voltage,
                                            nominal_dod=nominal_dod,
                                            nominal_lifetime=nominal_lifetime,
                                            c_max=c_max,
                                            temp_ambient=temp_ambient,
                                            v_max=v_max,
                                            v_min=v_min,
                                            elapsed_time=0.,
                                            electrical_state=electrical_state,
                                            thermal_state=thermal_state,
                                            soc_state=soc_state,
                                            aging_state=aging_state,
                                            soh=1.,
                                            iter=0,
                                            check_soh_every=check_soh_every)

        init_state = jax.tree.map(lambda leaf: jnp.array(leaf), init_state)

        return init_state

    @classmethod
    def _build_models(cls, models_settings, inits, sign_convention, temp_battery):
        electrical_state = None
        thermal_state = None
        aging_state = None

        for model_config in models_settings:
            if model_config['type'] == 'electrical':
                assert model_config['class_name'] == 'TheveninModel'
                assert model_config['use_fading'] == False
                electrical_state = TheveninModel.get_init_state(components=model_config['components'],
                                                                inits=inits,
                                                                sign_convention=sign_convention)

            elif model_config['type'] == 'thermal':
                assert model_config['class_name'] == 'R2CThermal'
                thermal_state = R2CThermalModel.get_init_state(model_config['components'], temp_battery)

            elif model_config['type'] == 'aging':
                assert model_config['class_name'] == 'BolunDropflowModel'
                #TODO AGGIUNGERE LIMITI ARRAY
                aging_state = BolunDropflowModel.get_init_state(components_setting=model_config['components'],
                                                                stress_models=model_config['stress_models'])

        return electrical_state, thermal_state, aging_state


    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step(cls, state: BessBolunDropflowState, i:float, dt:float) -> BessBolunDropflowState:
        new_electrical_state, v_out, _ = TheveninModel.step_current_driven(state.electrical_state, i, dt)

        new_soc_state, curr_soc = SOCModel.compute_soc(state.soc_state, i, dt, state.nominal_capacity)

        dissipated_heat = TheveninModel.compute_generated_heat(new_electrical_state)

        new_thermal_state, curr_temp = R2CThermalModel.compute_temp(state.thermal_state, q=dissipated_heat, i=i, T_amb=state.temp_ambient, dt=dt)

        new_aging_state, curr_soh = BolunDropflowModel.compute_soh(state.aging_state, curr_temp, state.temp_ambient, curr_soc, state.elapsed_time, state.iter % state.check_soh_every == 0)

        new_c_max = curr_soh * state.nominal_capacity

        new_state = state.replace(electrical_state=new_electrical_state,
                                  thermal_state=new_thermal_state,
                                  soc_state=new_soc_state,
                                  aging_state=new_aging_state,
                                  c_max=new_c_max,
                                  iter=state.iter+1)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def get_feasible_current(cls, state: BessBolunDropflowState, soc, dt):
        return SOCModel.get_feasible_current(state.soc_state, soc, state.c_max, dt)