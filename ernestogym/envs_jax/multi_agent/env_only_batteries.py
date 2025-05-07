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

    iter: int
    timeframe: int
    # is_rec_turn: bool


class RECEnv(MultiAgentEnv):
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = 60 * 60 * 24
    DAYS_PER_YEAR = 365.25

    def __init__(self, settings, battery_type):
        super().__init__(settings['num_battery_agents'])
        self.num_battery_agents = settings['num_battery_agents']
        # self.num_agents = self.num_battery_agents + 1

        self.num_passive_houses = settings['num_passive_houses']

        self.agents = [f'battery_agent_{i}' for i in range(self.num_battery_agents)]

        self.env_step = settings['step']

        assert len(settings['batteries']) == self.num_battery_agents

        batteries = []

        if battery_type == 'fading':
            self.BESS = bess_fading.BatteryEnergyStorageSystem
        elif battery_type == 'degrading':
            self.BESS = bess_degrading.BatteryEnergyStorageSystem
        elif battery_type == 'degrading_dropflow':
            self.BESS = bess_degrading_dropflow.BatteryEnergyStorageSystem
        else:
            raise ValueError(f'Unsupported battery aging: {battery_type}')

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

        self._termination = settings['termination']
        if self._termination['max_iterations'] is None:
            self._termination['max_iterations'] = jnp.inf

        self.trading_coeff = jnp.array(settings['reward']['trading_coeff'] if 'trading_coeff' in settings['reward'] else 0)
        self.op_cost_coeff = jnp.array(settings['reward']['operational_cost_coeff'] if 'operational_cost_coeff' in settings['reward'] else 0)
        self.deg_coeff = jnp.array(settings['reward']['degradation_coeff'] if 'degradation_coeff' in settings['reward'] else 0)
        self.clip_action_coeff = jnp.array(settings['reward']['clip_action_coeff'] if 'clip_action_coeff' in settings['reward'] else 0)
        self.glob_coeff = jnp.array(settings['reward']['glob_coeff'] if 'glob_coeff' in settings['reward'] else 0)
        self.glob_reward_distribution = settings.get('glob_reward_distribution', 'uniform')
        self.use_reward_normalization = settings['use_reward_normalization']

        self.return_dict_reward = settings['return_dict_reward']

        assert self.trading_coeff.shape == () or self.trading_coeff.shape == (self.num_battery_agents,)
        assert self.op_cost_coeff.shape == () or self.op_cost_coeff.shape == (self.num_battery_agents,)
        assert self.deg_coeff.shape == () or self.deg_coeff.shape == (self.num_battery_agents,)
        assert self.clip_action_coeff.shape == () or self.clip_action_coeff.shape == (self.num_battery_agents,)
        assert self.glob_coeff.shape == () or self.glob_coeff.shape == (self.num_battery_agents,)

        print('norm? ' + str(self.use_reward_normalization))

        ########################## OBSERVATION SPACES ##########################

        # self.observation_spaces = OrderedDict([(a, OrderedDict()) for a in self.agents])
        battery_obs_space = OrderedDict()

        self.obs_keys = ['temperature', 'soc', 'demand', 'generation', 'buying_price', 'selling_price']

        self.obs_is_sequence = {'temperature': True,
                                'soc': True,
                                'demand': True,
                                'generation': True,
                                'buying_price': True,
                                'selling_price': True}

        self.obs_is_local = {'temperature': True,
                             'soc': True,
                             'demand': True,
                             'generation': True,
                             'buying_price': True,
                             'selling_price': True}

        self.obs_is_normalizable = {'temperature': True,
                                    'soc': False,
                                    'demand': True,
                                    'generation': True,
                                    'buying_price': True,
                                    'selling_price': True}

        battery_obs_space['temperature'] = {'low': 250., 'high': 400.}
        battery_obs_space['soc'] = {'low': 0., 'high': 1.}
        battery_obs_space['demand'] = {'low': 0., 'high': jnp.inf}
        battery_obs_space['generation'] = {'low': 0., 'high': jnp.inf}
        battery_obs_space['buying_price'] = {'low': 0., 'high': jnp.inf}
        battery_obs_space['selling_price'] = {'low': 0., 'high': jnp.inf}

        if 'soh' in settings['battery_obs']:
            self.obs_keys.append('soh')
            battery_obs_space['soh'] = {'low': 0., 'high': 1.}
            self.obs_is_sequence['soh'] = True
            self.obs_is_local['soh'] = True
            self.obs_is_normalizable['soh'] = False

        if 'day_of_year' in settings['battery_obs']:
            self.obs_keys.append('sin_day_of_year')
            self.obs_keys.append('cos_day_of_year')
            battery_obs_space['sin_day_of_year'] = {'low': -1, 'high': 1}
            battery_obs_space['cos_day_of_year'] = {'low': -1, 'high': 1}
            self.obs_is_sequence['sin_day_of_year'] = False
            self.obs_is_sequence['cos_day_of_year'] = False
            self.obs_is_local['sin_day_of_year'] = True
            self.obs_is_local['cos_day_of_year'] = True
            self.obs_is_normalizable['sin_day_of_year'] = False
            self.obs_is_normalizable['cos_day_of_year'] = False

        if 'seconds_of_day' in settings['battery_obs']:
            self.obs_keys.append('sin_seconds_of_day')
            self.obs_keys.append('cos_seconds_of_day')
            battery_obs_space['sin_seconds_of_day'] = {'low': -1, 'high': 1}
            battery_obs_space['cos_seconds_of_day'] = {'low': -1, 'high': 1}
            self.obs_is_sequence['sin_seconds_of_day'] = False
            self.obs_is_sequence['cos_seconds_of_day'] = False
            self.obs_is_local['sin_seconds_of_day'] = True
            self.obs_is_local['cos_seconds_of_day'] = True
            self.obs_is_normalizable['sin_seconds_of_day'] = False
            self.obs_is_normalizable['cos_seconds_of_day'] = False

        # if 'energy_level' in settings['battery_obs']:
        #     self.obs_battery_agents_keys.append('energy_level')
        #     for i, a in enumerate(self.battery_agents):
        #         min_energy = batteries[i].nominal_capacity * batteries[i].soc_state.soc_min * batteries[i].v_max
        #         max_energy = batteries[i].nominal_capacity * batteries[i].soc_state.soc_max * batteries[i].v_min
        #         self.observation_spaces[a]['energy_level'] = spaces.Box(low=min_energy, high=max_energy, shape=(1,))


        if 'network_REC_plus' in settings['battery_obs']:
            self.obs_keys.append('network_REC_plus')
            battery_obs_space['network_REC_plus'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence['network_REC_plus'] = True
            self.obs_is_local['network_REC_plus'] = False
            self.obs_is_normalizable['network_REC_plus'] = True

        if 'network_REC_minus' in settings['battery_obs']:
            self.obs_keys.append('network_REC_minus')
            battery_obs_space['network_REC_minus'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence['network_REC_minus'] = True
            self.obs_is_local['network_REC_minus'] = False
            self.obs_is_normalizable['network_REC_minus'] = True

        if 'network_REC_diff' in settings['battery_obs']:
            self.obs_keys.append('network_REC_diff')
            battery_obs_space['network_REC_diff'] = {'low': -jnp.inf, 'high': jnp.inf}
            self.obs_is_sequence['network_REC_diff'] = True
            self.obs_is_local['network_REC_diff'] = False
            self.obs_is_normalizable['network_REC_diff'] = True

        if 'self_consumption_marginal_contribution' in settings['battery_obs']:
            self.obs_keys.append('self_consumption_marginal_contribution')
            battery_obs_space['self_consumption_marginal_contribution'] = {'low': 0, 'high': jnp.inf}
            self.obs_is_sequence['self_consumption_marginal_contribution'] = True
            self.obs_is_local['self_consumption_marginal_contribution'] = False
            self.obs_is_normalizable['self_consumption_marginal_contribution'] = True

        if 'rec_actions_prev_step' in settings['battery_obs']:
            self.obs_keys.append('rec_actions_prev_step')
            battery_obs_space['rec_actions_prev_step'] = {'low': 0, 'high': 1}
            self.obs_is_sequence['rec_actions_prev_step'] = True
            self.obs_is_local['rec_actions_prev_step'] = False
            self.obs_is_normalizable['rec_actions_prev_step'] = True

        self.observation_spaces = OrderedDict([(a, spaces.Dict({key: spaces.Box(battery_obs_space[key]['low'],
                                                                                battery_obs_space[key]['high'],
                                                                                shape=(1,))
                                                                for key in self.obs_keys}))
                                               for a in self.agents])

        print(self.obs_keys)

        self.i_max_action = self.BESS.get_feasible_current(battery_states, battery_states.soc_state.soc_min, dt=self.env_step)[0]
        self.i_min_action = self.BESS.get_feasible_current(battery_states, battery_states.soc_state.soc_max, dt=self.env_step)[1]

        self.action_spaces = {a: spaces.Box(self.i_min_action[i], self.i_max_action[i], shape=(1,)) for i, a in enumerate(self.agents)}

        self.init_state = EnvState(battery_states=battery_states,
                                   iter=0,
                                   timeframe=0,
                                   done=jnp.zeros(shape=(self.num_agents,), dtype=bool),
                                   step=-1,
                                   demands_battery_houses=demands_battery_houses,
                                   demands_passive_houses=demands_passive_houses)

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

        temperatures = state.battery_states.thermal_state.temp
        soc = state.battery_states.soc_state.soc
        balance_plus, balance_minus = self._calc_balances(state, past_shift=self.env_step)

        obs_array = {}

        for key in self.obs_keys:
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

        obs = {a: jax.tree.map(lambda x: x[i], obs_array) for i, a in enumerate(self.agents)}

        return obs

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

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
            reset_state: Optional[State] = None,
            stop_gradient_r_trad=False,
            stop_gradient_r_deg=False,
            stop_gradient_r_op=False,
            stop_gradient_r_clip=False,
            stop_gradient_r_glob=False,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions,
                                                                 stop_gradient_r_trad=stop_gradient_r_trad,
                                                                 stop_gradient_r_deg=stop_gradient_r_deg,
                                                                 stop_gradient_r_op=stop_gradient_r_op,
                                                                 stop_gradient_r_clip=stop_gradient_r_clip,
                                                                 stop_gradient_r_glob=stop_gradient_r_glob,
                                                                 )

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos


    def step_env(self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array],
                 stop_gradient_r_trad=False,
                 stop_gradient_r_deg=False,
                 stop_gradient_r_op=False,
                 stop_gradient_r_clip=False,
                 stop_gradient_r_glob=False,
                 ) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array([actions[a].flatten()[0] for a in self.agents])

        new_timeframe = state.timeframe + self.env_step

        i_max, i_min = jax.vmap(self.BESS.get_feasible_current, in_axes=(0, 0, None))(state.battery_states, state.battery_states.soc_state.soc, self.env_step)
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
        r_deg = self._calc_deg_reward(old_soh, new_battery_states.soh, new_battery_states.nominal_cost, self._termination['min_soh'])
        r_op = jnp.zeros_like(r_deg)

        # if stop_gradient_r_trad:
        #     r_trading = jax.lax.stop_gradient(r_trading)
        # if stop_gradient_r_clip:
        #     r_clipping = jax.lax.stop_gradient(r_clipping)
        # if stop_gradient_r_deg:
        #     r_deg = jax.lax.stop_gradient(r_deg)
        # if stop_gradient_r_op:
        #     r_op = jax.lax.stop_gradient(r_op)

        r_trading = jax.lax.stop_gradient(r_trading)
        r_clipping = jax.lax.stop_gradient(r_clipping)
        r_deg = jax.lax.stop_gradient(r_deg)
        r_op = jax.lax.stop_gradient(r_op)

        balance_plus, balance_minus = self._calc_balances(state)
        self_consumption = jnp.minimum(balance_plus, balance_minus)
        tot_incentives = self._calc_rec_incentives(state, self_consumption)
        r_glob = self._calc_glob_reward_distribution(tot_incentives)

        if stop_gradient_r_glob:
            r_glob = jax.lax.stop_gradient(r_glob)

        norm_r_trading, norm_r_op, norm_r_deg, norm_r_clipping, norm_r_glob = self._normalize_reward(state, new_battery_states, r_trading, r_op, r_deg, r_clipping, r_glob)
        weig_r_trading, weig_r_op, weig_r_deg, weig_r_clipping, weig_r_glob = (self.trading_coeff * norm_r_trading,
                                                                               self.op_cost_coeff * norm_r_op,
                                                                               self.deg_coeff * norm_r_deg,
                                                                               self.clip_action_coeff * norm_r_clipping,
                                                                               self.glob_coeff * norm_r_glob)

        if self.return_dict_reward:
            r_tot = {'r_trad': weig_r_trading,
                     'r_op': weig_r_op,
                     'r_deg': weig_r_deg,
                     'r_clipping': weig_r_clipping,
                     'r_glob': weig_r_glob}
        else:
            r_tot = weig_r_trading + weig_r_op + weig_r_deg + weig_r_clipping + weig_r_glob

        new_iteration = state.iter + 1

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

        rewards = {a: jax.tree.map(lambda x: x[i], r_tot) for i, a in enumerate(self.agents)}

        # rewards = {a: r_tot[i] for i, a in enumerate(self.agents)}

        dones_array = jnp.logical_or(truncated, terminated)
        dones = {a: dones_array[i] for i, a in enumerate(self.agents)}
        dones['__all__'] = jnp.any(dones_array)

        new_state = state.replace(battery_states=new_battery_states,
                                  iter=new_iteration,
                                  timeframe=new_timeframe,
                                  done=dones_array)

        info = {'soc': new_battery_states.soc_state.soc,
                'soh': new_battery_states.soh,
                'pure_reward': {'r_trad': r_trading,
                                'r_op': r_op,
                                'r_deg': r_deg,
                                'r_clipping': r_clipping,
                                'r_glob': r_glob},
                'norm_reward': {'r_trad': norm_r_trading,
                                'r_op': norm_r_op,
                                'r_deg': norm_r_deg,
                                'r_clipping': norm_r_clipping,
                                'r_glob': norm_r_glob},
                'weig_reward': {'r_trad': weig_r_trading,
                                'r_op': weig_r_op,
                                'r_deg': weig_r_deg,
                                'r_clipping': weig_r_clipping,
                                'r_glob': weig_r_glob},
                'r_tot': r_tot,
                'self_consumption': self_consumption,
                'balance_plus': balance_plus,
                'balance_minus': balance_minus,
                'tot_incentives': tot_incentives,
                'generations': generations,
                'demands': demands,
                'buy_prices': buying_prices,
                'sell_prices': selling_prices,
                'energy_to_batteries': new_state.battery_states.electrical_state.p}



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

    def _normalize_reward(self, state: EnvState, battery_states: BessState, r_trading, r_op, r_deg, r_clipping, r_glob):

        if self.use_reward_normalization:
            norm_r_trading = r_trading / jnp.maximum(self.generations_battery_houses.max * self.selling_prices_battery_houses.max,
                                                         state.demands_battery_houses.max * self.buying_prices_battery_houses.max)
            norm_r_op = r_op / (battery_states.nominal_cost + 1e-8)

            return norm_r_trading, norm_r_op, r_deg, r_clipping, r_glob

        else:
            return r_trading, r_op, r_deg, r_clipping, r_glob

    def _calc_rec_incentives(self, state: EnvState, self_consumption: float):
        valorization_part = self_consumption * self.valorization_incentive_coeff
        incentivizing_tariff_fixed = self_consumption * self.incentivizing_tariff_coeff

        incentivizing_tariff_variable = self_consumption *  jnp.minimum(self.incentivizing_tariff_max_variable, jnp.maximum(0, self.incentivizing_tariff_baseline_variable - BuyingPrice.get_buying_price(self.market, state.timeframe)))

        return valorization_part + incentivizing_tariff_fixed + incentivizing_tariff_variable

    def _calc_glob_reward_distribution(self, tot_incentives):
        tot_incentives_to_battery_agents = tot_incentives * self.num_battery_agents / (
                    self.num_battery_agents + self.num_passive_houses)

        if self.glob_reward_distribution == 'uniform':
            return jnp.full((self.num_battery_agents,), tot_incentives_to_battery_agents/self.num_battery_agents)
        else:
            raise ValueError(f'{self.glob_reward_distribution} is not a valid global reward distribution')