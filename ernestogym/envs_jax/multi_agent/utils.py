"""
This module primarily implements the `parameter_generator()` function which
generates the parameters dict for `EnergyStorageEnv`.
"""
import os.path
import random
from typing import List, NamedTuple
import yaml
from pint import UnitRegistry
from ernestogym.ernesto_jax.utils import read_csv
from ernestogym.ernesto_jax import read_yaml, validate_yaml_parameters

BATTERY_OPTIONS = "ernestogym/ernesto/data/battery/cell.yaml"
INPUT_VAR = 'current'     # 'power'/'current'/'voltage'

ECM = "ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml"
R2C_THERMAL = "ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml"
BOLUN_MODEL = "ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml"
WORLD = "ernestogym/envs/single_agent/world_fading.yaml"

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)


class WorldMetadata(NamedTuple):
    world_train: dict
    world_test: dict
    battery: dict
    electrical: dict
    thermal: dict
    aging: dict

def _get_world(num_battery_houses, demand, generation, ambient_temp, market, additional_battery_obs=None, additional_rec_obs=None, **kwargs):
    defaults = {
        'num_passive_houses': 0,
        'step': 3600,
        'termination': {'max_iterations': None, 'min_soh': 0.6},
        'reward': {
            'trading_coeff': 1,
            'operational_cost_coeff': 0,
            'degradation_coeff': 1,
            'clip_action_coeff': 1,
            'glob_coeff': 1
        },
        'aging_options': {
            'degradation': True,
            'fading': False
        },
        'use_reward_normalization': False,
        'valorization_incentive_coeff': 0.000008,
        'incentivizing_tariff_coeff': 0.00008,
        'incentivizing_tariff_max_variable': 0.00004,
        'incentivizing_tariff_baseline_variable': 0.00018,
        'fairness_coeff': 0.,
        'smoothing_factor_rec_actions': 0.99
    }

    world = {
        'num_battery_houses': num_battery_houses,
        'battery_observations': ['demand', 'generation', 'temperature', 'soc', 'day_of_year', 'seconds_of_day',
                                 'selling_price', 'buying_price'] + (additional_battery_obs if additional_battery_obs is not None else []),
        'rec_observations': ['demands_base_battery_houses', 'demands_battery_battery_houses',
                             'generations_base_battery_houses', 'mean_demands_base', 'mean_demands_batteries',
                             'mean_generations', 'day_of_year', 'seconds_of_day', 'network_REC_plus',
                             'network_REC_minus'] + (additional_rec_obs if additional_rec_obs is not None else []),
        'demand': demand,
        'generation': generation,
        'temp_amb': ambient_temp,
        'market': market
    }

    world.update({**defaults, **kwargs})

    return world


def get_world_metadata(num_battery_houses,
                       demand_train, generation_train, ambient_temp_train, market_train,
                       demand_test, generation_test, ambient_temp_test, market_test,
                       additional_battery_obs=None, additional_rec_obs=None,
                       pack_options='ernestogym/ernesto_jax/data/battery_new/pack_init_half_full_cheap.yaml',
                       electrical='ernestogym/ernesto_jax/data/battery_new/models/electrical/thevenin_pack.yaml',
                       thermal='ernestogym/ernesto_jax/data/battery_new/models/thermal/r2c_thermal_pack.yaml',
                       bolun='ernestogym/ernesto_jax/data/battery_new/models/aging/bolun_pack.yaml',
                       **world_kwargs):

    world_train = _get_world(num_battery_houses, demand_train, generation_train, ambient_temp_train, market_train, additional_battery_obs, additional_rec_obs, **world_kwargs)
    world_test = _get_world(num_battery_houses, demand_test, generation_test, ambient_temp_test, market_test, additional_battery_obs, additional_rec_obs, **world_kwargs)

    battery = read_yaml(pack_options, yaml_type='battery_options')
    elec = read_yaml(electrical, yaml_type='model')
    ther = read_yaml(thermal, yaml_type='model')
    aging = read_yaml(bolun, yaml_type='model')

    world_metadata = WorldMetadata(world_train=world_train,
                                   world_test=world_test,
                                   battery=battery,
                                   electrical=elec,
                                   thermal=ther,
                                   aging=aging)

    return world_metadata

def get_world_data(world_metadata:WorldMetadata, get_train=False, get_test=False):
    ret = ()
    if get_train:
        ret += (parameter_generator(battery_options=world_metadata.battery,
                                    world_options=world_metadata.world_train,
                                    electrical_model=world_metadata.electrical,
                                    thermal_model=world_metadata.thermal,
                                    aging_model=world_metadata.aging),)

    if get_test:
        ret += (parameter_generator(battery_options=world_metadata.battery,
                                    world_options=world_metadata.world_test,
                                    electrical_model=world_metadata.electrical,
                                    thermal_model=world_metadata.thermal,
                                    aging_model=world_metadata.aging),)

    if len(ret) == 1:
        ret = ret[0]

    return ret

def get_world_metadata_from_template(template_name:str, world_kwargs=None, template_folder_path:str='ernestogym/envs_jax/multi_agent/templates'):

    with open(os.path.join(template_folder_path, template_name + '.yaml'), 'r') as fin:
        template = yaml.safe_load(fin)

    world_kwargs_template = template.get('world_kwargs', {})

    if world_kwargs is not None:
        world_kwargs_template.update(world_kwargs)


    world_metadata = get_world_metadata(num_battery_houses=template['num_battery_houses'],
                                        demand_train=template['train']['demand'],
                                        generation_train=template['train']['generation'],
                                        ambient_temp_train=template['train']['ambient_temp'],
                                        market_train=template['train']['market'],
                                        demand_test=template['test']['demand'],
                                        generation_test=template['test']['generation'],
                                        ambient_temp_test=template['test']['ambient_temp'],
                                        market_test=template['test']['market'],
                                        additional_battery_obs=template['additional_battery_obs'],
                                        additional_rec_obs=template['additional_rec_obs'],
                                        pack_options=template['pack_options'],
                                        electrical=template['electrical'],
                                        thermal=template['thermal'],
                                        aging=template['aging'],
                                        **world_kwargs_template)

    return world_metadata


def parameter_generator(battery_options: dict,
                        world_options: dict,
                        electrical_model: dict,
                        thermal_model: dict,
                        aging_model: dict,
                        input_var: str = INPUT_VAR,
                        seed: int = 123,
                        ) -> dict:
    """
    Generates the parameters dict for `EnergyStorageEnv`.
    """

    num_battery_houses = world_options['num_battery_houses']
    num_passive_houses = world_options['num_passive_houses']

    battery_options['battery']['params'] = validate_yaml_parameters(battery_options['battery']['params'])

    batteries_params = [battery_options['battery']] * num_battery_houses

    electrical_models = [electrical_model] * num_battery_houses
    thermal_models = [thermal_model] * num_battery_houses

    aging_options = {'degradation': world_options['aging_options']['degradation'],
                     'fading': world_options['aging_options']['fading']}

    model_configs = []
    if aging_options['degradation']:
        aging_models = [aging_model] * num_battery_houses

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
    if isinstance(world_options['demand']['path'], str):
        demand = read_csv(world_options['demand']['path']).drop(columns=['delta_time'])
        demand_profiles = split_demand_between_houses(demand, num_battery_houses + num_passive_houses)
        battery_houses_demand_profiles = demand_profiles[:num_battery_houses]
        passive_houses_demand_profiles = demand_profiles[num_battery_houses:]
        battery_houses_demands = [demand[profiles] for profiles in battery_houses_demand_profiles]
        passive_houses_demands = [demand[profiles] for profiles in passive_houses_demand_profiles]
        battery_houses_demand_paths = [world_options['demand']['path']] * num_battery_houses
        passive_houses_demand_paths = [world_options['demand']['path']] * num_passive_houses
    elif isinstance(world_options['demand']['path'], dict):
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

        battery_houses_demands, battery_houses_demand_profiles, battery_houses_demand_paths = dict_demand(world_options['demand']['path']['battery_houses_demand'], num_battery_houses)
        if num_passive_houses > 0:
            passive_houses_demands, passive_houses_demand_profiles, passive_houses_demand_paths = dict_demand(world_options['demand']['path']['passive_houses_demand'], num_passive_houses)
        else:
            passive_houses_demands, passive_houses_demand_profiles, passive_houses_demand_paths = [], [], []
    else:
        raise TypeError('demand.path must be a dict or a str')

    if isinstance(world_options['generation']['path'], str):
        generation_data = read_csv(world_options['generation']['path'])['PV']
        battery_houses_generations = [generation_data] * num_battery_houses
        passive_houses_generations = [generation_data] * num_passive_houses
        battery_houses_generation_paths = [world_options['generation']['path']] * num_battery_houses
        passive_houses_generation_paths = [world_options['generation']['path']] * num_passive_houses
    elif isinstance(world_options['generation']['path'], dict):
        def dict_generation(value, num):
            if isinstance(value, str):
                generation_data = read_csv(value)['PV']
                generations = [generation_data] * num
                generation_paths = [value] * num
            elif isinstance(value, list):
                print('yaaaas2')
                generations = []
                generation_paths = []
                for path in value:
                    generation = read_csv(path)['PV']
                    generations.append(generation)
                    generation_paths.append(path)
            else:
                raise TypeError('demand.path.battery_houses must be a list or a str')

            return generations, generation_paths

        battery_houses_generations, battery_houses_generation_paths = dict_generation(world_options['generation']['path']['battery_houses_generation'], num_battery_houses)
        if num_passive_houses > 0:
            passive_houses_generations, passive_houses_generation_paths = dict_generation(world_options['generation']['path']['passive_houses_generation'], num_passive_houses)
        else:
            passive_houses_generations, passive_houses_generation_paths = [], []
    else:
        raise TypeError('demand.path must be a list or a str')


    market = read_csv(world_options['market']['path'])
    buying_price_data = market['ask']
    selling_price_data = market['bid']

    temp_data = read_csv(world_options['temp_amb']['path'])['temp_amb']

    params = {'num_battery_agents': num_battery_houses,
              'num_passive_houses': num_passive_houses,
              'batteries': batteries_params,
              'model_config': model_configs,
              'input_var': input_var,
              'demands_battery_houses': [{'data': battery_houses_demands[i],
                                          'timestep': world_options['demand']['timestep'],
                                          'demand_profiles': battery_houses_demand_profiles[i],
                                          'data_usage': world_options['demand']['data_usage'],
                                          'path': battery_houses_demand_paths[i]} for i in range(num_battery_houses)],
              'generations_battery_houses': [{'data': battery_houses_generations[i],
                                              'timestep': world_options['generation']['timestep'],
                                              'data_usage': world_options['generation']['data_usage'],
                                              'path': battery_houses_generation_paths[i]} for i in range(num_battery_houses)],
              'selling_prices_battery_houses': [{'data': selling_price_data,
                                                 'timestep': world_options['market']['timestep'],
                                                 'data_usage': world_options['market']['data_usage'],
                                                 'path': world_options['market']['path']}] * num_battery_houses,
              'buying_prices_battery_houses': [{'data': buying_price_data,
                                                'timestep': world_options['market']['timestep'],
                                                'data_usage': world_options['market']['data_usage'],
                                                'path': world_options['market']['path']}] * num_battery_houses,
              'temp_amb_battery_houses': [{'data': temp_data,
                                           'timestep': world_options['temp_amb']['timestep'],
                                           'data_usage': world_options['temp_amb']['data_usage'],
                                           'path': world_options['temp_amb']['path']}] * num_battery_houses,

              'demands_passive_houses': [{'data': passive_houses_demands[i],
                                          'timestep': world_options['demand']['timestep'],
                                          'demand_profiles': passive_houses_demand_profiles[i],
                                          'data_usage': world_options['demand']['data_usage'],
                                          'path': passive_houses_demand_paths[i]} for i in range(num_passive_houses)],
              'generations_passive_houses': [{'data': passive_houses_generations[i],
                                              'timestep': world_options['generation']['timestep'],
                                              'data_usage': world_options['generation']['data_usage'],
                                              'path': passive_houses_generation_paths[i]} for i in range(num_passive_houses)],
              'selling_prices_passive_houses': [{'data': selling_price_data,
                                                 'timestep': world_options['market']['timestep'],
                                                 'data_usage': world_options['market']['data_usage'],
                                                 'path': world_options['market']['path']}] * num_passive_houses,
              'buying_prices_passive_houses': [{'data': buying_price_data,
                                                'timestep': world_options['market']['timestep'],
                                                'data_usage': world_options['market']['data_usage'],
                                                'path': world_options['market']['path']}] * num_passive_houses,
              # 'temp_amb_passive_houses': [{'data': temp_data,
              #                              'timestep': world_settings['temp_amb']['timestep'],
              #                              'data_usage': world_settings['temp_amb']['data_usage'],
              #                              'path': world_settings['temp_amb']['path']}] * num_passive_houses,

              'market': {'data': read_csv(world_options['market']['path'])['ask'],
                         'timestep': world_options['market']['timestep'],
                         'data_usage': world_options['market']['data_usage']},

              'battery_obs': world_options['battery_observations'],
              'rec_obs': world_options['rec_observations'],

              'step': world_options['step'],
              'seed': seed if seed is not None else world_options['seed'],
              'aging_options': aging_options,
              'reward': world_options['reward'],
              'use_reward_normalization': world_options['use_reward_normalization'],
              'termination': {'max_iterations': world_options['termination']['max_iterations'],
                             'min_soh': world_options['termination']['min_soh']},

              'valorization_incentive_coeff': world_options['valorization_incentive_coeff'],
              'incentivizing_tariff_coeff': world_options['incentivizing_tariff_coeff'],
              'incentivizing_tariff_max_variable': world_options['incentivizing_tariff_max_variable'],
              'incentivizing_tariff_baseline_variable': world_options['incentivizing_tariff_baseline_variable'],
              'fairness_coeff': world_options['fairness_coeff'],
              'smoothing_factor_rec_actions': world_options['smoothing_factor_rec_actions']

    }

    return params
