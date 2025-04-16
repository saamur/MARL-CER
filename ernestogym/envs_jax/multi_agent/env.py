import chex
from flax import struct
from typing import Dict, Tuple, Optional
from collections import OrderedDict

import numpy as np

import jax
import jax.numpy as jnp
from jaxmarl.environments import State
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import jaxmarl.environments.spaces  as spaces

from functools import partial

from ernestogym.ernesto_jax.demand import Demand, DemandData
from ernestogym.ernesto_jax.generation import Generation
from ernestogym.ernesto_jax.market import BuyingPrice, SellingPrice
from ernestogym.ernesto_jax.ambient_temperature import AmbientTemperature

from ernestogym.ernesto_jax.energy_storage.bess import BessState
import ernestogym.ernesto_jax.energy_storage.bess_fading as bess_fading
import ernestogym.ernesto_jax.energy_storage.bess_degrading as bess_degrading
import ernestogym.ernesto_jax.energy_storage.bess_degrading_dropflow as bess_degrading_dropflow


@struct.dataclass
class EnvState(State):
    battery_states: BessState

    demands_battery_houses: DemandData
    demands_passive_houses: DemandData

    prev_actions_rec: jnp.array
    exp_avg_rev_actions_rec: jnp.array

    iter: int
    timeframe: int
    is_rec_turn: bool


class RECEnv(MultiAgentEnv):
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = 60 * 60 * 24
    DAYS_PER_YEAR = 365.25

    def __init__(self, settings, battery_type):
        super().__init__(settings['num_battery_agents'] + 1)
        self.num_battery_agents = settings['num_battery_agents']
        # self.num_agents = self.num_battery_agents + 1

        self.num_passive_houses = settings['num_passive_houses']

        self.battery_agents = [f'battery_agent_{i}' for i in range(self.num_battery_agents)]

        self.rec_agent = 'REC_agent'

        self.agents = self.battery_agents + [self.rec_agent]

        self.env_step = settings['step']

        assert len(settings['batteries']) == self.num_battery_agents

        batteries = []

        if battery_type == 'fading':
            self.BESS = bess_fading.BatteryEnergyStorageSystem
        elif battery_type == 'degrading':
            self.BESS = bess_degrading.BatteryEnergyStorageSystem
        if battery_type == 'degrading_dropflow':
            self.BESS = bess_degrading_dropflow.BatteryEnergyStorageSystem
        else:
            raise ValueError(f'Unsupported battery aging: {settings['aging_type']}')

        for i in range(self.num_battery_agents):
            batteries.append(self.BESS.get_init_state(models_config=settings['model_config'][i],
                                                      battery_options=settings['batteries'][i],
                                                      input_var=settings['input_var']))


        battery_states = jax.tree.map(lambda *vals: jnp.array(vals), *batteries)


        ########################## DEMAND, GENERATION AND PRICES ##########################

        def setup_demand_generation_prices(demand_list, generation_list, selling_price_list, buying_prices_list, temp_list, length):

            assert len(demand_list) == length
            assert len(generation_list) == length
            assert len(selling_price_list) == length

            dem_step = demand_list[0]['timestep']
            gen_step = generation_list[0]['timestep']
            buy_step = buying_prices_list[0]['timestep']
            sell_step = selling_price_list[0]['timestep']

            dem_matrices_raw = [jnp.array(dem['data'].to_numpy().T) for dem in demand_list]                 #num_battery_agents x num_profiles x length

            gen_d = [gen['data'].to_numpy() for gen in generation_list]                                     #num_battery_agents x length
            buy_d = [buy['data'].to_numpy() for buy in buying_prices_list]
            sell_d = [sell['data'].to_numpy() for sell in selling_price_list]

            if temp_list is not None:
                assert len(buying_prices_list) == length
                temp_step = temp_list[0]['timestep']
                temp_d = [temp['data'].to_numpy() for temp in temp_list]


            max_length = min(dem_matrices_raw[0].shape[1] * dem_step,
                             len(gen_d[0]) * gen_step,
                             len(buy_d[0]) * buy_step,
                             len(sell_d[0]) * sell_step)

            if temp_list is not None:
                max_length = min (max_length, len(temp_d[0]) * temp_step)

            dem_matrices = jnp.array([[Demand.build_demand_array(dem_prof, in_timestep=dem_step, out_timestep=self.env_step, max_length=max_length)
                                       for dem_prof in matrix_agent]
                                      for matrix_agent in dem_matrices_raw])

            demands = [Demand.build_demand_data(agent_matrix[0], self.env_step) for agent_matrix in dem_matrices]
            generations = [Generation.build_generation_data(data, in_timestep=gen_step, out_timestep=self.env_step, max_length=max_length) for data in gen_d]
            selling_prices = [SellingPrice.build_selling_price_data(data, in_timestep=sell_step, out_timestep=self.env_step, max_length=max_length) for data in sell_d]
            buying_prices = [BuyingPrice.build_buying_price_data(data, in_timestep=buy_step, out_timestep=self.env_step, max_length=max_length) for data in buy_d]

            ret = (dem_matrices,
                   jax.tree.map(lambda *vals: jnp.array(vals), *demands),
                   jax.tree.map(lambda *vals: jnp.array(vals), *generations),
                   jax.tree.map(lambda *vals: jnp.array(vals), *selling_prices),
                   jax.tree.map(lambda *vals: jnp.array(vals), *buying_prices))

            if temp_list is not None:
                temperatures = [AmbientTemperature.build_generation_data(data, in_timestep=temp_step, out_timestep=self.env_step, max_length=max_length) for data in temp_d]
                ret += (jax.tree.map(lambda *vals: jnp.array(vals), *temperatures),)

            return ret


        (self.dem_matrices_battery_houses,
         demands_battery_houses,
         self.generations_battery_houses,
         self.selling_prices_battery_houses,
         self.buying_prices_battery_houses,
         self.temp_ambient) = setup_demand_generation_prices(settings['demands_battery_houses'],
                                                             settings['generations_battery_houses'],
                                                             settings['selling_prices_battery_houses'],
                                                             settings['buying_prices_battery_houses'],
                                                             settings['temp_amb_battery_houses'],
                                                             self.num_battery_agents)

        if self.num_passive_houses > 0:
            (self.dem_matrices_passive_houses,
             demands_passive_houses,
             self.generations_passive_houses,
             self.selling_prices_passive_houses,
             self.buying_prices_passive_houses) = setup_demand_generation_prices(settings['demands_passive_houses'],
                                                                                 settings['generations_passive_houses'],
                                                                                 settings['selling_prices_passive_houses'],
                                                                                 settings['buying_prices_passive_houses'],
                                                                                 None,
                                                                                 self.num_passive_houses)
        else:
            demands_passive_houses = 0

        self.market = BuyingPrice.build_buying_price_data(jnp.array(settings['market']['data'].to_numpy()), settings['market']['timestep'], self.env_step, settings['market']['timestep'] * len(settings['market']['data']), False)

        self.valorization_incentive_coeff = settings['valorization_incentive_coeff']
        self.incentivizing_tariff_coeff = settings['incentivizing_tariff_coeff']
        self.incentivizing_tariff_max_variable = settings['incentivizing_tariff_max_variable']
        self.incentivizing_tariff_baseline_variable = settings['incentivizing_tariff_baseline_variable']
        self.fairness_coeff = settings['fairness_coeff']

        self._termination = settings['termination']
        if self._termination['max_iterations'] is None:
            self._termination['max_iterations'] = jnp.inf

        self.trading_coeff = jnp.array(settings['reward']['trading_coeff'] if 'trading_coeff' in settings['reward'] else 0)
        self.op_cost_coeff = jnp.array(settings['reward']['operational_cost_coeff'] if 'operational_cost_coeff' in settings['reward'] else 0)
        self.deg_coeff = jnp.array(settings['reward']['degradation_coeff'] if 'degradation_coeff' in settings['reward'] else 0)
        self.clip_action_coeff = jnp.array(settings['reward']['clip_action_coeff'] if 'clip_action_coeff' in settings['reward'] else 0)
        self.glob_coeff = jnp.array(settings['reward']['glob_coeff'] if 'glob_coeff' in settings['reward'] else 0)
        self.use_reward_normalization = settings['use_reward_normalization']

        assert self.trading_coeff.shape == () or self.trading_coeff.shape == (self.num_battery_agents,)
        assert self.op_cost_coeff.shape == () or self.op_cost_coeff.shape == (self.num_battery_agents,)
        assert self.deg_coeff.shape == () or self.deg_coeff.shape == (self.num_battery_agents,)
        assert self.clip_action_coeff.shape == () or self.clip_action_coeff.shape == (self.num_battery_agents,)
        assert self.glob_coeff.shape == () or self.glob_coeff.shape == (self.num_battery_agents,)

        print('norm? ' + str(self.use_reward_normalization))

        self.smoothing_factor_rec_actions = settings['smoothing_factor_rec_actions']

        ########################## OBSERVATION SPACES ##########################

        # self.observation_spaces = OrderedDict([(a, OrderedDict()) for a in self.agents])
        self.battery_obs_space = OrderedDict()

        self.obs_battery_agents_keys = ['temperature', 'soc', 'demand', 'generation', 'buying_price', 'selling_price']

        self.obs_is_sequence_battery = {'temperature': True,
                                        'soc': True,
                                        'demand': True,
                                        'generation': True,
                                        'buying_price': True,
                                        'selling_price': True}

        self.obs_is_local_battery = {'temperature': True,
                                     'soc': True,
                                     'demand': True,
                                     'generation': True,
                                     'buying_price': True,
                                     'selling_price': True}

        self.obs_is_normalizable_battery = {'temperature': True,
                                            'soc': False,
                                            'demand': True,
                                            'generation': True,
                                            'buying_price': True,
                                            'selling_price': True}

        self.battery_obs_space['temperature'] = {'low': 250., 'high': 400.}
        self.battery_obs_space['soc'] = {'low': 0., 'high': 1.}
        self.battery_obs_space['demand'] = {'low': 0., 'high': jnp.inf}
        self.battery_obs_space['generation'] = {'low': 0., 'high': jnp.inf}
        self.battery_obs_space['buying_price'] = {'low': 0., 'high': jnp.inf}
        self.battery_obs_space['selling_price'] = {'low': 0., 'high': jnp.inf}

        # for a in self.battery_agents:
        #     self.observation_spaces[a]['temperature'] = spaces.Box(low=250., high=400., shape=(1,))
        #     self.observation_spaces[a]['soc'] = spaces.Box(low=0., high=1., shape=(1,))
        #     self.observation_spaces[a]['demand'] = spaces.Box(low=0., high=1., shape=(1,))
        #     self.observation_spaces[a]['generation'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
        #     self.observation_spaces[a]['buying_price'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
        #     self.observation_spaces[a]['selling_price'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))

        # Add optional 'State of Health' in observation space
        if 'soh' in settings['battery_obs']:
            # spaces['soh'] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.obs_battery_agents_keys.append('soh')
            self.battery_obs_space['soh'] = {'low': 0., 'high': 1.}
            self.obs_is_sequence_battery['soh'] = True
            self.obs_is_local_battery['soh'] = True
            self.obs_is_normalizable_battery['soh'] = False

        if 'day_of_year' in settings['battery_obs']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.obs_battery_agents_keys.append('sin_day_of_year')
            self.obs_battery_agents_keys.append('cos_day_of_year')
            self.battery_obs_space['sin_day_of_year'] = {'low': -1, 'high': 1}
            self.battery_obs_space['cos_day_of_year'] = {'low': -1, 'high': 1}
            self.obs_is_sequence_battery['sin_day_of_year'] = False
            self.obs_is_sequence_battery['cos_day_of_year'] = False
            self.obs_is_local_battery['sin_day_of_year'] = True
            self.obs_is_local_battery['cos_day_of_year'] = True
            self.obs_is_normalizable_battery['sin_day_of_year'] = False
            self.obs_is_normalizable_battery['cos_day_of_year'] = False

        if 'seconds_of_day' in settings['battery_obs']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.obs_battery_agents_keys.append('sin_seconds_of_day')
            self.obs_battery_agents_keys.append('cos_seconds_of_day')
            self.battery_obs_space['sin_seconds_of_day'] = {'low': -1, 'high': 1}
            self.battery_obs_space['cos_seconds_of_day'] = {'low': -1, 'high': 1}
            self.obs_is_sequence_battery['sin_seconds_of_day'] = False
            self.obs_is_sequence_battery['cos_seconds_of_day'] = False
            self.obs_is_local_battery['sin_seconds_of_day'] = True
            self.obs_is_local_battery['cos_seconds_of_day'] = True
            self.obs_is_normalizable_battery['sin_seconds_of_day'] = False
            self.obs_is_normalizable_battery['cos_seconds_of_day'] = False

        # if 'energy_level' in settings['battery_obs']:
        #     self.obs_battery_agents_keys.append('energy_level')
        #     for i, a in enumerate(self.battery_agents):
        #         min_energy = batteries[i].nominal_capacity * batteries[i].soc_state.soc_min * batteries[i].v_max
        #         max_energy = batteries[i].nominal_capacity * batteries[i].soc_state.soc_max * batteries[i].v_min
        #         self.observation_spaces[a]['energy_level'] = spaces.Box(low=min_energy, high=max_energy, shape=(1,))


        if 'network_REC_plus' in settings['battery_obs']:
            self.obs_battery_agents_keys.append('network_REC_plus')
            self.battery_obs_space['network_REC_plus'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence_battery['network_REC_plus'] = True
            self.obs_is_local_battery['network_REC_plus'] = False
            self.obs_is_normalizable_battery['network_REC_plus'] = True

        if 'network_REC_minus' in settings['battery_obs']:
            self.obs_battery_agents_keys.append('network_REC_minus')
            self.battery_obs_space['network_REC_minus'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence_battery['network_REC_minus'] = True
            self.obs_is_local_battery['network_REC_minus'] = False
            self.obs_is_normalizable_battery['network_REC_minus'] = True

        if 'network_REC_diff' in settings['battery_obs']:
            self.obs_battery_agents_keys.append('network_REC_diff')
            self.battery_obs_space['network_REC_diff'] = {'low': -jnp.inf, 'high': jnp.inf}
            self.obs_is_sequence_battery['network_REC_diff'] = True
            self.obs_is_local_battery['network_REC_diff'] = False
            self.obs_is_normalizable_battery['network_REC_diff'] = True

        if 'self_consumption_marginal_contribution' in settings['battery_obs']:
            self.obs_battery_agents_keys.append('self_consumption_marginal_contribution')
            self.battery_obs_space['self_consumption_marginal_contribution'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence_battery['self_consumption_marginal_contribution'] = True
            self.obs_is_local_battery['self_consumption_marginal_contribution'] = False
            self.obs_is_normalizable_battery['self_consumption_marginal_contribution'] = True

        if 'rec_actions_prev_step' in settings['battery_obs']:
            self.obs_battery_agents_keys.append('rec_actions_prev_step')
            self.battery_obs_space['rec_actions_prev_step'] = {'low': 0, 'high': 1}
            self.obs_is_sequence_battery['rec_actions_prev_step'] = True
            self.obs_is_local_battery['rec_actions_prev_step'] = False
            self.obs_is_normalizable_battery['rec_actions_prev_step'] = True


        # indices = np.argsort(np.logical_not(obs_is_sequence))
        #
        # self.obs_battery_agents_keys = [self.obs_battery_agents_keys[i] for i in indices]
        # self.obs_is_normalizable_battery = [self.obs_is_normalizable_battery[i] for i in indices]
        # self.obs_is_local_battery = [self.obs_is_local_battery[i] for i in indices]
        # self.obs_is_sequence_battery = [obs_is_sequence[i] for i in indices]

        # self.num_battery_obs_sequences = np.sum(obs_is_sequence)

        # self._obs_battery_agents_idx = {key: i for i, key in enumerate(self.obs_battery_agents_keys)}

        self.observation_spaces = OrderedDict([(a, spaces.Dict({key: spaces.Box(self.battery_obs_space[key]['low'],
                                                                                self.battery_obs_space[key]['high'],
                                                                                shape=(1,))
                                                                 for key in self.obs_battery_agents_keys}))
                                               for a in self.battery_agents])

        # self.observation_spaces = OrderedDict([(a, spaces.Box(jnp.array([self.battery_obs_space[key]['low'] for key in self.obs_battery_agents_keys]),
        #                                                       jnp.array([self.battery_obs_space[key]['high'] for key in self.obs_battery_agents_keys]),
        #                                                       shape=(len(self.obs_battery_agents_keys),)))
        #                                        for a in self.battery_agents])

        print(self.obs_battery_agents_keys)


        self.obs_rec_keys = ['demands_base_battery_houses', 'demands_battery_battery_houses', 'generations_battery_houses']

        self.obs_is_sequence_rec = {'demands_base_battery_houses':True,
                                    'demands_battery_battery_houses':True,
                                    'generations_battery_houses':True}

        self.obs_is_local_rec = {'demands_base_battery_houses':True,
                                 'demands_battery_battery_houses':True,
                                 'generations_battery_houses':True}

        self.obs_is_normalizable_rec = {'demands_base_battery_houses': True,
                                        'demands_battery_battery_houses': True,
                                        'generations_battery_houses': True}


        rec_obs_space = {'demands_base_battery_houses': spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,)),
                         'demands_battery_battery_houses': spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,)),
                         'generations_battery_houses': spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,))}

        if self.num_passive_houses > 0:
            if 'demands_passive_houses' in settings['rec_obs']:
                self.obs_rec_keys.append('demands_passive_houses')
                rec_obs_space['demands_passive_houses'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_passive_houses,))
                self.obs_is_sequence_rec['demands_passive_houses'] = True
                self.obs_is_local_rec['demands_passive_houses'] = True
                self.obs_is_normalizable_rec['demands_passive_houses'] = True
            if 'generations_passive_houses' in settings['rec_obs']:
                self.obs_rec_keys.append('generations_passive_houses')
                rec_obs_space['generations_passive_houses'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_passive_houses,))
                self.obs_is_sequence_rec['generations_passive_houses'] = True
                self.obs_is_local_rec['generations_passive_houses'] = True
                self.obs_is_normalizable_rec['generations_passive_houses'] = True

        if 'tot_demands_base' in settings['rec_obs']:
            self.obs_rec_keys.append('tot_demands_base')
            rec_obs_space['tot_demands_base'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.obs_is_sequence_rec['tot_demands_base'] = True
            self.obs_is_local_rec['tot_demands_base'] = False
            self.obs_is_normalizable_rec['tot_demands_base'] = True

        if 'tot_demands_batteries' in settings['rec_obs']:
            self.obs_rec_keys.append('tot_demands_batteries')
            rec_obs_space['tot_demands_batteries'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.obs_is_sequence_rec['tot_demands_batteries'] = True
            self.obs_is_local_rec['tot_demands_batteries'] = False
            self.obs_is_normalizable_rec['tot_demands_batteries'] = True

        if 'tot_generations' in settings['rec_obs']:
            self.obs_rec_keys.append('tot_generations')
            rec_obs_space['tot_generations'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.obs_is_sequence_rec['tot_generations'] = True
            self.obs_is_local_rec['tot_generations'] = False
            self.obs_is_normalizable_rec['tot_generations'] = True

        if 'mean_demands_base' in settings['rec_obs']:
            self.obs_rec_keys.append('mean_demands_base')
            rec_obs_space['mean_demands_base'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.obs_is_sequence_rec['mean_demands_base'] = True
            self.obs_is_local_rec['mean_demands_base'] = False
            self.obs_is_normalizable_rec['mean_demands_base'] = True

        if 'mean_demands_batteries' in settings['rec_obs']:
            self.obs_rec_keys.append('mean_demands_batteries')
            rec_obs_space['mean_demands_batteries'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.obs_is_sequence_rec['mean_demands_batteries'] = True
            self.obs_is_local_rec['mean_demands_batteries'] = False
            self.obs_is_normalizable_rec['mean_demands_batteries'] = True

        if 'mean_generations' in settings['rec_obs']:
            self.obs_rec_keys.append('mean_generations')
            rec_obs_space['mean_generations'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.obs_is_sequence_rec['mean_generations'] = True
            self.obs_is_local_rec['mean_generations'] = False
            self.obs_is_normalizable_rec['mean_generations'] = True

        if 'day_of_year' in settings['rec_obs']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.obs_rec_keys.append('sin_day_of_year')
            self.obs_rec_keys.append('cos_day_of_year')
            rec_obs_space['sin_day_of_year'] = spaces.Box(low=-1., high=1., shape=(1,))
            rec_obs_space['cos_day_of_year'] = spaces.Box(low=-1., high=1., shape=(1,))
            self.obs_is_sequence_rec.update({'sin_day_of_year': False, 'cos_day_of_year': False})
            self.obs_is_local_rec.update({'sin_day_of_year': False, 'cos_day_of_year': False})
            self.obs_is_normalizable_rec.update({'sin_day_of_year': False, 'cos_day_of_year': False})
        if 'seconds_of_day' in settings['rec_obs']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.obs_rec_keys.append('sin_seconds_of_day')
            self.obs_rec_keys.append('cos_seconds_of_day')
            rec_obs_space['sin_seconds_of_day'] = spaces.Box(low=-1., high=1., shape=(1,))
            rec_obs_space['cos_seconds_of_day'] = spaces.Box(low=-1., high=1., shape=(1,))
            self.obs_is_sequence_rec.update({'sin_seconds_of_day': False, 'cos_seconds_of_day': False})
            self.obs_is_local_rec.update({'sin_seconds_of_day': False, 'cos_seconds_of_day': False})
            self.obs_is_normalizable_rec.update({'sin_seconds_of_day': False, 'cos_seconds_of_day': False})

        if 'network_REC_plus' in settings['rec_obs']:
            self.obs_rec_keys.append('network_REC_plus')
            rec_obs_space['network_REC_plus'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence_rec['network_REC_plus'] = True
            self.obs_is_local_rec['network_REC_plus'] = False
            self.obs_is_normalizable_rec['network_REC_plus'] = True

        if 'network_REC_minus' in settings['rec_obs']:
            self.obs_rec_keys.append('network_REC_minus')
            rec_obs_space['network_REC_minus'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence_rec['network_REC_minus'] = True
            self.obs_is_local_rec['network_REC_minus'] = False
            self.obs_is_normalizable_rec['network_REC_minus'] = True

        if 'network_REC_diff' in settings['rec_obs']:
            self.obs_rec_keys.append('network_REC_diff')
            rec_obs_space['network_REC_diff'] = {'low': -jnp.inf, 'high': jnp.inf}
            self.obs_is_sequence_rec['network_REC_diff'] = True
            self.obs_is_local_rec['network_REC_diff'] = False
            self.obs_is_normalizable_rec['network_REC_diff'] = True

        if 'rec_actions_prev_step' in settings['rec_obs']:
            self.obs_rec_keys.append('rec_actions_prev_step')
            rec_obs_space['rec_actions_prev_step'] = spaces.Box(low=0., high=1., shape=(self.num_battery_agents,))
            self.obs_is_sequence_rec['rec_actions_prev_step'] = True
            self.obs_is_local_rec['rec_actions_prev_step'] = True
            self.obs_is_normalizable_rec['rec_actions_prev_step'] = False

        if 'exponential_average_rec_actions_prev_step' in settings['rec_obs']:
            self.obs_rec_keys.append('exponential_average_rec_actions_prev_step')
            rec_obs_space['exponential_average_rec_actions_prev_step'] = spaces.Box(low=0., high=1., shape=(self.num_battery_agents,))
            self.obs_is_sequence_rec['exponential_average_rec_actions_prev_step'] = False
            self.obs_is_local_rec['exponential_average_rec_actions_prev_step'] = True
            self.obs_is_normalizable_rec['exponential_average_rec_actions_prev_step'] = False

        if 'battery_agents_marginal_contribution' in settings['rec_obs']:
            self.obs_rec_keys.append('battery_agents_marginal_contribution')
            rec_obs_space['battery_agents_marginal_contribution'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,))
            self.obs_is_sequence_rec['battery_agents_marginal_contribution'] = True
            self.obs_is_local_rec['battery_agents_marginal_contribution'] = True
            self.obs_is_normalizable_rec['battery_agents_marginal_contribution'] = True

        # self._obs_rec_keys = tuple(self._obs_rec_keys)

        print(self.obs_rec_keys)

        self.observation_spaces[self.rec_agent] = spaces.Dict(rec_obs_space)


        self.i_max_action = self.BESS.get_feasible_current(battery_states, battery_states.soc_state.soc_min, dt=self.env_step)[0]
        self.i_min_action = self.BESS.get_feasible_current(battery_states, battery_states.soc_state.soc_max, dt=self.env_step)[1]

        self.action_spaces = {a: spaces.Box(self.i_min_action[i], self.i_max_action[i], shape=(1,)) for i, a in enumerate(self.battery_agents)}
        self.action_spaces[self.rec_agent] = spaces.Box(0., 1., shape=(self.num_battery_agents,))

        self.init_state = EnvState(battery_states=battery_states,
                                   iter=0,
                                   is_rec_turn=False,
                                   timeframe=0,
                                   done=jnp.zeros(shape=(self.num_agents,), dtype=bool),
                                   step=-1,
                                   demands_battery_houses=demands_battery_houses,
                                   demands_passive_houses=demands_passive_houses,
                                   prev_actions_rec=jnp.ones(self.num_battery_agents)/self.num_battery_agents,
                                   exp_avg_rev_actions_rec=jnp.ones(self.num_battery_agents)/self.num_battery_agents)

    @partial(jax.vmap, in_axes=(None, 0, None))
    def _get_generations(self, gen_data, timestep):
        return Generation.get_generation(gen_data, timestep)

    @partial(jax.vmap, in_axes=(None, 0, None))
    def _get_demands(self, dem_data, timestep):
        return Demand.get_demand(dem_data, timestep)

    @partial(jax.vmap, in_axes=(None, 0, None))
    def _get_selling_prices(self, sell_price_data, timestep):
        return SellingPrice.get_selling_price(sell_price_data, timestep)

    @partial(jax.vmap, in_axes=(None, 0, None))
    def _get_buying_prices(self, buy_price_data, timestep):
        return BuyingPrice.get_buying_price(buy_price_data, timestep)

    @partial(jax.vmap, in_axes=(None, 0, None))
    def _get_temperatures(self, temperature_data, timestep):
        return AmbientTemperature.get_amb_temperature(temperature_data, timestep)

    def _calc_balances(self, state: EnvState, past_shift=0):
        demands_batteries = self._get_demands(state.demands_battery_houses, state.timeframe-past_shift)
        generations_batteries = self._get_generations(self.generations_battery_houses, state.timeframe-past_shift)

        power_batteries = state.battery_states.electrical_state.p

        balance_battery_houses = generations_batteries - demands_batteries - power_batteries

        if self.num_passive_houses > 0:
            demands_passive_houses = self._get_demands(state.demands_passive_houses, state.timeframe-past_shift)
            generations_passive_houses = self._get_generations(self.generations_passive_houses, state.timeframe-past_shift)
            balance_passive_houses = generations_passive_houses - demands_passive_houses

            balances = jnp.concat([balance_battery_houses, balance_passive_houses])
        else:
            balances = balance_battery_houses

        balance_plus = jnp.where(balances >= 0, balances, 0).sum()
        balance_minus = jnp.where(balances < 0, balances, 0).sum()

        return balance_plus, -balance_minus

    def _calc_marginal_contributions(self, state: EnvState, past_shift=0):
        demands_batteries = self._get_demands(state.demands_battery_houses, state.timeframe - past_shift)
        generations_batteries = self._get_generations(self.generations_battery_houses, state.timeframe - past_shift)

        power_batteries = state.battery_states.electrical_state.p

        balance_battery_houses = generations_batteries - demands_batteries - power_batteries

        balance_battery_houses_plus = jnp.where(balance_battery_houses >= 0, balance_battery_houses, 0)
        balance_battery_houses_minus = -jnp.where(balance_battery_houses < 0, balance_battery_houses, 0)

        balance_plus = balance_battery_houses_plus.sum()
        balance_minus = balance_battery_houses_minus.sum()

        if self.num_passive_houses > 0:
            demands_passive_houses = self._get_demands(state.demands_passive_houses, state.timeframe-past_shift)
            generations_passive_houses = self._get_generations(self.generations_passive_houses, state.timeframe-past_shift)
            balance_passive_houses = generations_passive_houses - demands_passive_houses

            balance_passive_houses_plus = jnp.where(balance_passive_houses >= 0, balance_passive_houses, 0)
            balance_passive_houses_minus = -jnp.where(balance_passive_houses < 0, balance_passive_houses, 0)

            balance_plus += balance_passive_houses_plus.sum()
            balance_minus += balance_passive_houses_minus.sum()

        marginal_balance_plus = balance_plus - balance_battery_houses_plus
        marginal_balance_minus = balance_minus - balance_battery_houses_minus
        marginal_contribution = jnp.minimum(balance_plus, balance_minus) - jnp.minimum(marginal_balance_plus, marginal_balance_minus)

        return marginal_contribution

    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        demands_batteries = self._get_demands(state.demands_battery_houses, state.timeframe)
        generations_batteries = self._get_generations(self.generations_battery_houses, state.timeframe)
        buying_price_batteries = self._get_buying_prices(self.buying_prices_battery_houses, state.timeframe)
        selling_price_batteries = self._get_selling_prices(self.selling_prices_battery_houses, state.timeframe)

        def batteries_turn():
            temperatures = state.battery_states.thermal_state.temp
            soc = state.battery_states.soc_state.soc
            balance_plus, balance_minus = self._calc_balances(state, past_shift=self.env_step)

            obs_array = {}

            for key in self.obs_battery_agents_keys:
                match key:
                    case 'temperature':
                        obs_array['temperature'] = temperatures
                    case 'soc':
                        obs_array['soc'] = soc
                    case 'soh':
                        obs_array['soh'] = state.battery_states.soh
                    case 'demand':
                        obs_array['demand'] = demands_batteries
                    case 'generation':
                        obs_array['generation'] = generations_batteries
                    case 'buying_price':
                        obs_array['buying_price'] = buying_price_batteries
                    case 'selling_price':
                        obs_array['selling_price'] = selling_price_batteries
                    case 'sin_day_of_year':
                        obs_array['sin_day_of_year'] = jnp.full(shape=(self.num_battery_agents,),
                                                 fill_value=jnp.sin(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe))
                    case 'cos_day_of_year':
                        obs_array['cos_day_of_year'] = jnp.full(shape=(self.num_battery_agents,),
                                                 fill_value=jnp.cos(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe))
                    case 'sin_seconds_of_day':
                        obs_array['sin_seconds_of_day'] = jnp.full(shape=(self.num_battery_agents,), fill_value=jnp.sin(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe))
                    case 'cos_seconds_of_day':
                        obs_array['cos_seconds_of_day'] = jnp.full(shape=(self.num_battery_agents,), fill_value=jnp.cos(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe))
                    case 'network_REC_plus':
                        obs_array['network_REC_plus'] = jnp.full(shape=(self.num_battery_agents,), fill_value=balance_plus)
                    case 'network_REC_minus':
                        obs_array['network_REC_minus'] = jnp.full(shape=(self.num_battery_agents,), fill_value=balance_minus)
                    case 'network_REC_diff':
                        obs_array['network_REC_diff'] = jnp.full(shape=(self.num_battery_agents,), fill_value=balance_plus-balance_minus)
                    case 'self_consumption_marginal_contribution':
                        obs_array['self_consumption_marginal_contribution'] = self._calc_marginal_contributions(state)
                    case 'rec_actions_prev_step':
                        obs_array['rec_actions_prev_step'] = state.prev_actions_rec

            obs = {a: jax.tree.map(lambda x: x[i], obs_array) for i, a in enumerate(self.battery_agents)}

            rec_obs = {'demands_base_battery_houses': jnp.zeros(self.num_battery_agents),
                       'demands_battery_battery_houses': jnp.zeros(self.num_battery_agents),
                       'generations_battery_houses': jnp.zeros(self.num_battery_agents)}

            if self.num_passive_houses > 0:
                if 'demands_passive_houses' in self.obs_rec_keys:
                    rec_obs['demands_passive_houses'] = jnp.zeros(self.num_passive_houses)
                if 'generations_passive_houses' in self.obs_rec_keys:
                    rec_obs['generations_passive_houses'] = jnp.zeros(self.num_passive_houses)

            if 'rec_actions_prev_step' in self.obs_rec_keys:
                rec_obs['rec_actions_prev_step'] = jnp.zeros(self.num_battery_agents)

            if 'exponential_average_rec_actions_prev_step' in self.obs_rec_keys:
                rec_obs['exponential_average_rec_actions_prev_step'] = jnp.zeros(self.num_battery_agents)

            if 'battery_agents_marginal_contribution' in self.obs_rec_keys:
                rec_obs['battery_agents_marginal_contribution'] = jnp.zeros(self.num_battery_agents)

            for o in [key for key in self.obs_rec_keys if not self.obs_is_local_rec[key]]:
                rec_obs[o] = 0.
            obs[self.rec_agent] = rec_obs

            return obs

        def rec_turn():
            # obs_battery_agents = jnp.zeros((len(self.obs_battery_agents_keys),))

            obs = {a: {key: 0. for key in self.obs_battery_agents_keys} for a in self.battery_agents}

            rec_obs = {'demands_base_battery_houses': demands_batteries,
                       'demands_battery_battery_houses': state.battery_states.electrical_state.p,
                       'generations_battery_houses': generations_batteries}

            balance_plus, balance_minus = self._calc_balances(state)

            for key in self.obs_rec_keys:
                match key:
                    case 'tot_demands_base':
                        rec_obs['tot_demands_base'] = demands_batteries.sum()
                    case 'tot_demands_batteries':
                        rec_obs['tot_demands_batteries'] = state.battery_states.electrical_state.p.sum()
                    case 'tot_generations':
                        rec_obs['tot_generations'] = generations_batteries.sum()

                    case 'mean_demands_base':
                        rec_obs['mean_demands_base'] = demands_batteries.mean()
                    case 'mean_demands_batteries':
                        rec_obs['mean_demands_batteries'] = state.battery_states.electrical_state.p.mean()
                    case 'mean_generations':
                        rec_obs['mean_generations'] = generations_batteries.mean()

                    case 'sin_seconds_of_day':
                        rec_obs['sin_seconds_of_day'] = jnp.sin(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe)
                    case 'cos_seconds_of_day':
                        rec_obs['cos_seconds_of_day'] = jnp.cos(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe)
                    case 'sin_day_of_year':
                        rec_obs['sin_day_of_year'] = jnp.sin(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe)
                    case 'cos_day_of_year':
                        rec_obs['cos_day_of_year'] = jnp.cos(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe)

                    case 'network_REC_plus':
                        rec_obs['network_REC_plus'] = balance_plus
                    case 'network_REC_minus':
                        rec_obs['network_REC_minus'] = balance_minus
                    case 'network_REC_diff':
                        rec_obs['network_REC_diff'] = balance_plus - balance_minus

                    case 'rec_actions_prev_step':
                        rec_obs['rec_actions_prev_step'] = state.prev_actions_rec
                    case 'exponential_average_rec_actions_prev_step':
                        rec_obs['exponential_average_rec_actions_prev_step'] = state.exp_avg_rev_actions_rec
                    case 'battery_agents_marginal_contribution':
                        rec_obs['battery_agents_marginal_contribution'] = self._calc_marginal_contributions(state)

            if self.num_passive_houses > 0:
                passive_demands = self._get_demands(state.demands_passive_houses, state.timeframe)
                passive_generation = self._get_generations(self.generations_passive_houses, state.timeframe)
                if 'demands_passive_houses' in self.obs_rec_keys:
                    rec_obs['demand_passive_houses'] = passive_demands
                if 'generations_passive_houses' in self.obs_rec_keys:
                    rec_obs['generations_passive_houses'] = passive_generation
                if 'tot_demands_base' in self.obs_rec_keys:
                    rec_obs['tot_demands_base'] += passive_demands.sum()
                if 'tot_generations' in self.obs_rec_keys:
                    rec_obs['tot_generations'] += passive_generation.sum()
                if 'mean_demands_base' in self.obs_rec_keys:
                    rec_obs['mean_demands_base'] = (rec_obs['mean_demands_base'] * self.num_battery_agents + passive_demands.sum()) / (self.num_battery_agents + self.num_passive_houses)
                if 'mean_generations' in self.obs_rec_keys:
                    rec_obs['mean_generations'] = (rec_obs['mean_generations'] * self.num_battery_agents + passive_generation.sum()) / (self.num_battery_agents + self.num_passive_houses)

            obs[self.rec_agent] = rec_obs

            return obs

        return jax.lax.cond(state.is_rec_turn, rec_turn, batteries_turn)

    def reset(self, key: chex.PRNGKey, profile_index=-1) -> Tuple[Dict[str, chex.Array], EnvState]:
        state = self.init_state
        key, key_ = jax.random.split(key)
        profiles_indices = jax.lax.cond(profile_index == -1,
                                        lambda : jax.random.choice(key_, self.dem_matrices_battery_houses.shape[1], shape=(self.num_battery_agents,)),
                                        lambda : jnp.full(shape=(self.num_battery_agents,), fill_value=profile_index%self.dem_matrices_battery_houses.shape[1]))

        demands = jax.vmap(Demand.build_demand_data, in_axes=(0, None))(self.dem_matrices_battery_houses[jnp.arange(self.num_battery_agents), profiles_indices],
                                                                        self.env_step)
        state = state.replace(demands_battery_houses=demands)

        if self.num_passive_houses > 0:
            key, key_ = jax.random.split(key)
            profiles_indices = jax.lax.cond(profile_index == -1,
                                            lambda: jax.random.choice(key_, self.dem_matrices_passive_houses.shape[1],
                                                                      shape=(self.num_passive_houses,)),
                                            lambda: jnp.full(shape=(self.num_passive_houses,),
                                                             fill_value=profile_index %
                                                                        self.dem_matrices_passive_houses.shape[1]))

            demands = jax.vmap(Demand.build_demand_data, in_axes=(0, None))(self.dem_matrices_passive_houses[jnp.arange(self.num_passive_houses), profiles_indices],
                                                                            self.env_step)
            state.replace(demands_passive_houses=demands)

        return self.get_obs(state), state

    def step_env(self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        return jax.lax.cond(state.is_rec_turn,
                            self.step_rec,
                            self.step_batteries,
                            state, actions)

    def step_rec(self, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:

        balance_plus, balance_minus = self._calc_balances(state)

        self_consumption = jnp.minimum(balance_plus, balance_minus)

        tot_incentives = self._calc_rec_incentives(state, self_consumption)

        rec_reward = self._calc_rec_reward(self_consumption, actions[self.rec_agent])

        tot_incentives_to_battery_agents = tot_incentives * self.num_battery_agents / (self.num_battery_agents + self.num_passive_houses)

        r_glob = tot_incentives_to_battery_agents * actions[self.rec_agent]
        weig_r_glob = self.glob_coeff * r_glob

        terminated = state.battery_states.soh <= self._termination['min_soh']

        truncated = jnp.logical_or(state.iter >= self._termination['max_iterations'],
                                   jnp.logical_or(jnp.logical_or(jax.vmap(Demand.is_run_out_of_data, in_axes=(0, None))(
                                       state.demands_battery_houses, state.timeframe),
                                                                 jax.vmap(Generation.is_run_out_of_data,
                                                                          in_axes=(0, None))(
                                                                     self.generations_battery_houses, state.timeframe)),
                                                  jnp.logical_or(
                                                      jax.vmap(BuyingPrice.is_run_out_of_data, in_axes=(0, None))(
                                                          self.buying_prices_battery_houses, state.timeframe),
                                                      jax.vmap(SellingPrice.is_run_out_of_data, in_axes=(0, None))(
                                                          self.selling_prices_battery_houses, state.timeframe))),
                                   )


        rewards = {a: r_glob[i] for i, a in enumerate(self.battery_agents)}
        rewards[self.rec_agent] = rec_reward

        dones_array = jnp.logical_or(truncated, terminated)
        done_rec = jnp.any(dones_array)

        new_exp_avg_rec_actions_prev_step = self.smoothing_factor_rec_actions * state.exp_avg_rev_actions_rec + (1-self.smoothing_factor_rec_actions) * actions[self.rec_agent]

        new_state = state.replace(is_rec_turn=False,
                                  done=jnp.concat([dones_array, done_rec[jnp.newaxis]]),
                                  prev_actions_rec=actions[self.rec_agent],
                                  exp_avg_rev_actions_rec=new_exp_avg_rec_actions_prev_step)

        dones = {a: dones_array[i] for i, a in enumerate(self.battery_agents)}
        dones[self.rec_agent] = done_rec
        dones['__all__'] = jnp.any(dones_array)         #It makes sense in our case to use any and not all

        info = {'soc': jnp.zeros(self.num_battery_agents),
                'soh': jnp.zeros(self.num_battery_agents),
                'pure_reward': {'r_trad': jnp.zeros(self.num_battery_agents),
                                'r_op': jnp.zeros(self.num_battery_agents),
                                'r_deg': jnp.zeros(self.num_battery_agents),
                                'r_clipping': jnp.zeros(self.num_battery_agents),
                                'r_glob': r_glob},
                'norm_reward': {'r_trad': jnp.zeros(self.num_battery_agents),
                                'r_op': jnp.zeros(self.num_battery_agents),
                                'r_deg': jnp.zeros(self.num_battery_agents),
                                'r_clipping': jnp.zeros(self.num_battery_agents),
                                'r_glob': r_glob},
                'weig_reward': {'r_trad': jnp.zeros(self.num_battery_agents),
                                'r_op': jnp.zeros(self.num_battery_agents),
                                'r_deg': jnp.zeros(self.num_battery_agents),
                                'r_clipping': jnp.zeros(self.num_battery_agents),
                                'r_glob': weig_r_glob},
                'r_tot': weig_r_glob,
                # 'r_glob': r_glob,
                'self_consumption': self_consumption,
                'balance_plus': balance_plus,
                'balance_minus': balance_minus,
                'tot_incentives': tot_incentives,
                'rec_reward': rec_reward,
                'generations': jnp.zeros(self.num_battery_agents),
                'demands': jnp.zeros(self.num_battery_agents),
                'buy_prices': jnp.zeros(self.num_battery_agents),
                'sell_prices': jnp.zeros(self.num_battery_agents),
                'energy_to_batteries': jnp.zeros(self.num_battery_agents)}

        return self.get_obs(new_state), new_state, rewards, dones, info


    def step_batteries(self, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array([actions[a].flatten()[0] for a in self.battery_agents])

        new_timeframe = state.timeframe + self.env_step
        last_v = state.battery_states.electrical_state.v
        i_max, i_min = jax.vmap(self.BESS.get_feasible_current, in_axes=(0, 0, None))(state.battery_states, state.battery_states.soc_state.soc, self.env_step)

        # jax.debug.print('soc {x}', x=state.battery_states.soc_state.soc, ordered=True)
        #
        # jax.debug.print('act {x}', x=actions, ordered=True)
        #
        # jax.debug.print('min {x}, max {y}', x=i_min, y=i_max, ordered=True)

        i_to_apply = jnp.clip(actions, i_min, i_max)

        old_soh = state.battery_states.soh

        t_amb = self._get_temperatures(self.temp_ambient, new_timeframe)

        new_battery_states = jax.vmap(self.BESS.step, in_axes=(0, 0, None, 0))(state.battery_states, i_to_apply, self.env_step, t_amb)

        to_load = new_battery_states.electrical_state.p

        demands = self._get_demands(state.demands_battery_houses, new_timeframe)
        generations = self._get_generations(self.generations_battery_houses, new_timeframe)

        to_trade = generations - demands - to_load


        buying_prices = self._get_buying_prices(self.buying_prices_battery_houses, new_timeframe)
        selling_prices = self._get_selling_prices(self.selling_prices_battery_houses, new_timeframe)

        r_trading = jnp.minimum(0, to_trade) * buying_prices + jnp.maximum(0, to_trade) * selling_prices

        r_clipping = -jnp.square(actions - i_to_apply)

        # jax.debug.print('cl {x}, sq {y}', x=r_clipping, y=-jnp.square(actions - i_to_apply), ordered=True)

        r_deg = self._calc_deg_reward(old_soh, new_battery_states.soh, new_battery_states.nominal_cost, self._termination['min_soh'])

        # r_op = self._calc_op_reward(new_battery_states.nominal_cost,
        #                             new_battery_states.nominal_capacity * new_battery_states.nominal_voltage / 1000,
        #                             new_battery_states.c_max * new_battery_states.nominal_voltage / 1000,
        #                             new_battery_states.nominal_dod,
        #                             new_battery_states.nominal_lifetime,
        #                             new_battery_states.nominal_voltage,
        #                             new_battery_states.electrical_state.p,
        #                             new_battery_states.electrical_state.r0,
        #                             new_battery_states.electrical_state.rc.resistance,
        #                             new_battery_states.soc_state.soc,
        #                             new_battery_states.electrical_state.p <= 0)

        r_op = jnp.zeros_like(r_deg)

        norm_r_trading, norm_r_op, norm_r_deg, norm_r_clipping = self._normalize_reward(state, new_battery_states, r_trading, r_op, r_deg, r_clipping)
        weig_r_trading, weig_r_op, weig_r_deg, weig_r_clipping = (self.trading_coeff * norm_r_trading, self.op_cost_coeff * norm_r_op,
                                                                  self.deg_coeff * norm_r_deg, self.clip_action_coeff * norm_r_clipping)

        r_tot = weig_r_trading + weig_r_op + weig_r_deg + weig_r_clipping

        new_iteration = state.iter + 1

        # terminated = new_battery_states.soh <= self._termination['min_soh']

        # truncated = jnp.logical_or(new_iteration >= self._termination['max_iterations'],
        #                            jnp.logical_or(jnp.logical_or(jax.vmap(Demand.is_run_out_of_data, in_axes=(0, None))(state.demands_battery_houses, new_timeframe),
        #                                                          jax.vmap(Generation.is_run_out_of_data, in_axes=(0, None))(self.generations_battery_houses, new_timeframe)),
        #                                           jnp.logical_or(jax.vmap(BuyingPrice.is_run_out_of_data, in_axes=(0, None))(self.buying_prices_battery_houses, new_timeframe),
        #                                                          jax.vmap(SellingPrice.is_run_out_of_data, in_axes=(0, None))(self.selling_prices_battery_houses, new_timeframe))),
        #                            )

        new_state = state.replace(battery_states=new_battery_states,
                                  iter=new_iteration,
                                  timeframe=new_timeframe,
                                  is_rec_turn=True)

        rewards = {a: r_tot[i] for i, a in enumerate(self.battery_agents)}
        rewards[self.rec_agent] = jnp.array(0.)

        info = {'soc': new_battery_states.soc_state.soc,
                'soh': new_battery_states.soh,
                'pure_reward': {'r_trad': r_trading,
                                'r_op': r_op,
                                'r_deg': r_deg,
                                'r_clipping': r_clipping,
                                'r_glob': jnp.zeros(self.num_battery_agents)},
                'norm_reward': {'r_trad': norm_r_trading,
                                'r_op': norm_r_op,
                                'r_deg': norm_r_deg,
                                'r_clipping': norm_r_clipping,
                                'r_glob': jnp.zeros(self.num_battery_agents)},
                'weig_reward': {'r_trad': weig_r_trading,
                                'r_op': weig_r_op,
                                'r_deg': weig_r_deg,
                                'r_clipping': weig_r_clipping,
                                'r_glob': jnp.zeros(self.num_battery_agents)},
                'r_tot': r_tot,
                # 'r_glob': jnp.zeros(self.num_battery_agents),
                'self_consumption': 0.,
                'balance_plus': 0.,
                'balance_minus': 0.,
                'tot_incentives': 0.,
                'rec_reward': 0.,
                'generations': generations,
                'demands': demands,
                'buy_prices': buying_prices,
                'sell_prices': selling_prices,
                'energy_to_batteries': new_state.battery_states.electrical_state.p}

        # dones_array = jnp.logical_or(truncated, terminated)

        # dones = {a: dones_array[i] for i, a in enumerate(self.battery_agents)}
        # dones[self.rec_agent] = False
        # dones['__all__'] = jnp.all(dones_array)
        dones = {a: False for i, a in enumerate(self.battery_agents)}
        dones[self.rec_agent] = False
        dones['__all__'] = False

        return self.get_obs(new_state), new_state, rewards, dones, info


    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def _calc_deg_reward(self, old_soh, curr_soh, replacement_cost, soh_limit):

        delta_soh = jnp.abs(old_soh - curr_soh)
        soh_cost = delta_soh * replacement_cost / (1 - soh_limit)
        return -soh_cost

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    def _calc_op_reward(self, replacement_cost: float,
                     C_rated: float,
                     C: float,
                     DoD_rated: float,
                     L_rated: float,
                     v_rated: float,
                     p: float,
                     r: float,
                     K_rated: float,
                     soc: float,
                     is_discharging: bool
                     ) -> float:

        # To prevent division by zero error
        soc = jnp.where(soc == 0, 1e-6, soc)

        # Coefficient c_avai = c_bat
        c_bat = replacement_cost / (C_rated * DoD_rated * (0.9 * L_rated - 0.1))

        # P_loss depending on P charged or discharged
        h_bat = jax.lax.cond(is_discharging,
                             lambda : jnp.abs(p) + (1 * (r + K_rated / soc) / v_rated ** 2 * p ** 2 +
                                                    1 * C * K_rated * (1 - soc) / (soc * v_rated ** 2) * p),
                             lambda : (1 * (r + K_rated / (0.9 - soc)) / v_rated ** 2 * p ** 2 +
                                       1 * C * K_rated * (1 - soc) / (soc * v_rated ** 2) * p))

        # Dividing by 1e3 to convert because it is in €/kWh, to get the cost in €/Wh
        op_cost_term = c_bat * h_bat / 1e3

        return - op_cost_term

    def _normalize_reward(self, state: EnvState, battery_states: BessState, r_trading, r_op, r_deg, r_clipping):

        if self.use_reward_normalization:
            norm_r_trading = r_trading / jnp.maximum(self.generations_battery_houses.max * self.selling_prices_battery_houses.max,
                                                         state.demands_battery_houses.max * self.buying_prices_battery_houses.max)
            norm_r_op = r_op / (battery_states.nominal_cost + 1e-8)

            return norm_r_trading, norm_r_op, r_deg, r_clipping

        else:
            return r_trading, r_op, r_deg, r_clipping


    def _calc_rec_reward(self, self_consumption, actions):
        return self_consumption + self.fairness_coeff * jnp.var(actions)


    def _calc_rec_incentives(self, state: EnvState, self_consumption: float):
        valorization_part = self_consumption * self.valorization_incentive_coeff
        incentivizing_tariff_fixed = self_consumption * self.incentivizing_tariff_coeff

        incentivizing_tariff_variable = self_consumption *  jnp.minimum(self.incentivizing_tariff_max_variable, jnp.maximum(0, self.incentivizing_tariff_baseline_variable - BuyingPrice.get_buying_price(self.market, state.timeframe)))

        return valorization_part + incentivizing_tariff_fixed + incentivizing_tariff_variable