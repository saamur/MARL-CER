"""
This module primarily implements the `parameter_generator()` function which
generates the parameters dict for `EnergyStorageEnv`.
"""
import random
from typing import List
import yaml
from pint import UnitRegistry
from ernestogym.ernesto_jax.utils import read_csv
from ernestogym.ernesto_jax import read_yaml, validate_yaml_parameters
from os import listdir

BATTERY_OPTIONS = "ernestogym/ernesto/data/battery/cell.yaml"
INPUT_VAR = 'power'     # 'power'/'current'/'voltage'

ECM = "ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml"
R2C_THERMAL = "ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml"
BOLUN_MODEL = "ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml"
WORLD = "ernestogym/envs/single_agent/world_fading.yaml"

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)


def parameter_generator(battery_options: str= BATTERY_OPTIONS,
                        world_options: str = WORLD,
                        num_battery_houses: int = None,
                        num_passive_houses: int = None,
                        input_var: str = INPUT_VAR,
                        electrical_model: str = ECM,
                        thermal_model: str = R2C_THERMAL,
                        aging_model: str = BOLUN_MODEL,
                        use_degradation: bool = None,
                        use_fading: bool = None,
                        step: int = None,
                        random_battery_init: bool = None,
                        random_data_init: bool = None,
                        seed: int = 123,
                        max_iterations: int = None,
                        min_soh: float = None,
                        reward_coeff: dict[str, float] = None,
                        use_reward_normalization: bool = None,
                        bypass_yaml_schema: bool = False,
                        spread_factor: float = 1.0,
                        replacement_cost: float = 3000.0,
                        ) -> dict:
    """
    Generates the parameters dict for `EnergyStorageEnv`.
    """
    with open(world_options, "r") as fin:
        world_settings = yaml.safe_load(fin)

    num_battery_houses = num_battery_houses if num_battery_houses is not None else world_settings['num_battery_houses']
    num_passive_houses = num_battery_houses if num_passive_houses is not None else world_settings['num_passive_houses']

    # Battery parameters retrieved with ErNESTO APIs.


    battery = read_yaml(battery_options, yaml_type='battery_options', bypass_check=bypass_yaml_schema)
    battery['battery']['params'] = validate_yaml_parameters(battery['battery']['params'])

    if replacement_cost is not None:
        battery['battery']['params']['nominal_cost'] = replacement_cost

    batteries_params = [battery['battery']] * num_battery_houses

    elec = read_yaml(electrical_model, yaml_type='model', bypass_check=bypass_yaml_schema)
    electrical_models = [elec] * num_battery_houses

    ther = read_yaml(thermal_model, yaml_type='model', bypass_check=bypass_yaml_schema)
    thermal_models = [ther] * num_battery_houses

    aging_options = {'degradation': use_degradation if use_degradation is not None else world_settings['aging_options']['degradation'],
                     'fading': use_fading if use_fading is not None else world_settings['aging_options']['fading']}

    model_configs = []
    if aging_options['degradation']:
        aging = read_yaml(aging_model, yaml_type='model', bypass_check=bypass_yaml_schema)
        aging_models = [aging] * num_battery_houses

        for i in range(num_battery_houses):
            model_configs.append([electrical_models[i], thermal_models[i], aging_models[i]])
    else:
        for i in range(num_battery_houses):
            model_configs.append([electrical_models[i], thermal_models[i]])

    def split_demand_between_houses(df, num):
        demand_profiles_names = df.columns.tolist()

        random.shuffle(demand_profiles_names)

        num_profiles_each_house = len(demand_profiles_names) // num

        demand_profiles = []
        for start in range(0, len(demand_profiles_names), num_profiles_each_house):
            demand_profiles.append(demand_profiles_names[start:start + num_profiles_each_house])
            start += num_profiles_each_house

        return demand_profiles

    random.seed(seed)
    if isinstance(world_settings['demand']['path'], str):
        demand = read_csv(world_settings['demand']['path']).drop(columns=['delta_time'])
        demand_profiles = split_demand_between_houses(demand, num_battery_houses + num_passive_houses)
        battery_houses_demand_profiles = demand_profiles[:num_battery_houses]
        passive_houses_demand_profiles = demand_profiles[num_battery_houses:]
        battery_houses_demands = [demand[profiles] for profiles in battery_houses_demand_profiles]
        passive_houses_demands = [demand[profiles] for profiles in passive_houses_demand_profiles]
        battery_houses_demand_paths = [world_settings['demand']['path']] * num_battery_houses
        passive_houses_demand_paths = [world_settings['demand']['path']] * num_passive_houses
    elif isinstance(world_settings['demand']['path'], dict):
        def dict_demand(value, num):
            if isinstance(value, str):
                demand = read_csv(value).drop(columns=['delta_time'])
                demand_profiles = split_demand_between_houses(demand, num)
                demands = [demand[profiles] for profiles in demand_profiles]
                demand_paths = [value] * num
            elif isinstance(value, list):
                print('yaaaas')
                demand_profiles = []
                demands = []
                demand_paths = []
                for path in value:
                    demand = read_csv(path).drop(columns=['delta_time'])
                    demands.append(demand)
                    demand_profiles.append(demand.columns.tolist())
                    demand_paths.append(path)
            else:
                raise TypeError('demand.path.battery_houses must be a list or a str')

            return demands, demand_profiles, demand_paths

        battery_houses_demands, battery_houses_demand_profiles, battery_houses_demand_paths = dict_demand(world_settings['demand']['path']['battery_houses_demand'], num_battery_houses)
        if num_passive_houses > 0:
            passive_houses_demands, passive_houses_demand_profiles, passive_houses_demand_paths = dict_demand(world_settings['demand']['path']['passive_houses_demand'], num_passive_houses)
        else:
            passive_houses_demands, passive_houses_demand_profiles, passive_houses_demand_paths = [], [], []
    else:
        raise TypeError('demand.path must be a list or a str')



    generation_data = read_csv(world_settings['generation']['path'])['PV']
    market = read_csv(world_settings['market']['path'])
    buying_price_data = market['ask']
    selling_price_data = market['bid']

    temp_data = read_csv(world_settings['temp_amb']['path'])['temp_amb']

    params = {'num_battery_agents': num_battery_houses,
              'num_passive_houses': num_passive_houses,
              'batteries': batteries_params,
              'model_config': model_configs,
              'input_var': input_var,
              'demands_battery_houses': [{'data': battery_houses_demands[i],
                                          'timestep': world_settings['demand']['timestep'],
                                          'demand_profiles': battery_houses_demand_profiles[i],
                                          'data_usage': world_settings['demand']['data_usage'],
                                          'path': battery_houses_demand_paths[i]} for i in range(num_battery_houses)],
              'generations_battery_houses': [{'data': generation_data,
                                              'timestep': world_settings['generation']['timestep'],
                                              'data_usage': world_settings['generation']['data_usage'],
                                              'path': world_settings['generation']['path']}] * num_battery_houses,
              'selling_prices_battery_houses': [{'data': selling_price_data,
                                                 'timestep': world_settings['market']['timestep'],
                                                 'data_usage': world_settings['market']['data_usage'],
                                                 'path': world_settings['market']['path']}] * num_battery_houses,
              'buying_prices_battery_houses': [{'data': buying_price_data,
                                                'timestep': world_settings['market']['timestep'],
                                                'data_usage': world_settings['market']['data_usage'],
                                                'path': world_settings['market']['path']}] * num_battery_houses,
              'temp_amb_battery_houses': [{'data': temp_data,
                                           'timestep': world_settings['temp_amb']['timestep'],
                                           'data_usage': world_settings['temp_amb']['data_usage'],
                                           'path': world_settings['temp_amb']['path']}] * num_battery_houses,

              'demands_passive_houses': [{'data': passive_houses_demands[i],
                                          'timestep': world_settings['demand']['timestep'],
                                          'demand_profiles': passive_houses_demand_profiles[i],
                                          'data_usage': world_settings['demand']['data_usage'],
                                          'path': passive_houses_demand_paths[i]} for i in range(num_passive_houses)],
              'generations_passive_houses': [{'data': generation_data,
                                              'timestep': world_settings['generation']['timestep'],
                                              'data_usage': world_settings['generation']['data_usage'],
                                              'path': world_settings['generation']['path']}] * num_passive_houses,
              'selling_prices_passive_houses': [{'data': selling_price_data,
                                                 'timestep': world_settings['market']['timestep'],
                                                 'data_usage': world_settings['market']['data_usage'],
                                                 'path': world_settings['market']['path']}] * num_passive_houses,
              'buying_prices_passive_houses': [{'data': buying_price_data,
                                                'timestep': world_settings['market']['timestep'],
                                                'data_usage': world_settings['market']['data_usage'],
                                                'path': world_settings['market']['path']}] * num_passive_houses,
              # 'temp_amb_passive_houses': [{'data': temp_data,
              #                              'timestep': world_settings['temp_amb']['timestep'],
              #                              'data_usage': world_settings['temp_amb']['data_usage'],
              #                              'path': world_settings['temp_amb']['path']}] * num_passive_houses,

              'market': {'data': read_csv(world_settings['market']['path'])['ask'],
                         'timestep': world_settings['market']['timestep'],
                         'data_usage': world_settings['market']['data_usage'],
                         'spread_factor': spread_factor},

              'battery_obs': world_settings['battery_observations'],
              'rec_obs': world_settings['rec_observations'],

              'step': step if step is not None else world_settings['step'],
              'seed': seed if seed is not None else world_settings['seed'],
              'aging_options': aging_options,
              'reward': reward_coeff if reward_coeff is not None else world_settings['reward'],
              'use_reward_normalization': use_reward_normalization if use_reward_normalization is not None else world_settings['use_reward_normalization'],
              'termination': {'max_iterations': max_iterations if max_iterations is not None else world_settings['termination']['max_iterations'],
                             'min_soh': min_soh if min_soh is not None else world_settings['termination']['min_soh']},

              'valorization_incentive_coeff': world_settings['valorization_incentive_coeff'],
              'incentivizing_tariff_coeff': world_settings['incentivizing_tariff_coeff'],
              'incentivizing_tariff_max_variable': world_settings['incentivizing_tariff_max_variable'],
              'incentivizing_tariff_baseline_variable': world_settings['incentivizing_tariff_baseline_variable'],
              'fairness_coeff': world_settings['fairness_coeff']

    }

    return params
