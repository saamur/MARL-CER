import chex
from flax import struct
from typing import Dict, Tuple
from collections import OrderedDict

import jax
import jax.numpy as jnp
from jaxmarl.environments import State
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import jaxmarl.environments.spaces  as spaces

from functools import partial

from ernestogym.ernesto_jax.demand import Demand
from ernestogym.ernesto_jax.generation import Generation
from ernestogym.ernesto_jax.market import BuyingPrice, SellingPrice

from ernestogym.ernesto_jax.energy_storage.bess import BessState
import ernestogym.ernesto_jax.energy_storage.bess_fading as bess_fading
import ernestogym.ernesto_jax.energy_storage.bess_degrading as bess_degrading
import ernestogym.ernesto_jax.energy_storage.bess_degrading_dropflow as bess_degrading_dropflow


@struct.dataclass
class EnvState(State):
    battery_states: BessState

    iter: int
    timeframe: int
    is_rec_turn: bool


class RECEnv(MultiAgentEnv):
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = 60 * 60 * 24
    DAYS_PER_YEAR = 365.25

    def __init__(self, settings):
        self.num_battery_agents = settings['num_battery_agents']
        self.num_agents = self.num_battery_agents + 1

        self.num_passive_houses = settings['num_passive_houses']

        self.battery_agents = [f'battery_agent_{i}' for i in range(self.num_battery_agents)]

        self.rec_agent = 'REC_agent'

        self.agents = self.battery_agents + [self.rec_agent]

        self.env_step = settings['step']

        assert len(settings['batteries']) == self.num_battery_agents

        batteries = []

        if settings['aging_type'] == 'fading':
            self.BESS_class = bess_fading.BatteryEnergyStorageSystem
        elif settings['aging_type'] == 'degrading':
            self.BESS_class = bess_degrading.BatteryEnergyStorageSystem
        if settings['aging_type'] == 'degrading_dropflow':
            self.BESS_class = bess_degrading_dropflow.BatteryEnergyStorageSystem
        else:
            raise ValueError(f'Unsupported battery aging: {settings['aging_type']}')

        for b in settings['batteries']:
            batteries.append(self.BESS_class.get_init_state(models_config=b['model_config'],
                                                                 battery_options=b['battery_options'],
                                                                 input_var=settings['input_var']))


        battery_states = jax.tree.map(lambda *vals: jnp.array(vals), *batteries)


        ########################## DEMAND, GENERATION AND PRICES ##########################

        def setup_demand_generation_prices(demand_list, generation_list, selling_price_list, buying_prices_list, length):

            assert len(demand_list) == length
            assert len(generation_list) == length
            assert len(selling_price_list) == length
            assert len(buying_prices_list) == length

            demands_data = [d['data'] for d in demand_list]
            demands_timestep = [d['timestep'] for d in demand_list]
            generations_data = [g['data'] for g in generation_list]
            generations_timestep = [g['timestep'] for g in generation_list]
            selling_prices_data = [s['data'] for s in selling_price_list]
            selling_prices_timestep = [s['timestep'] for s in selling_price_list]
            buying_prices_data = [b['data'] for b in buying_prices_list]
            buying_prices_timestep = [b['timestep'] for b in buying_prices_list]
            buying_prices_circularity = [b['circular'] for b in buying_prices_list]

            max_length = max([len(data) * ts for data, ts in zip(demands_data + generations_data +
                                                                 selling_prices_data + [buying_prices_data[i] for i in range(length) if not buying_prices_circularity[i]],
                                                                 demands_timestep + generations_timestep +
                                                                 selling_prices_timestep + [buying_prices_timestep[i] for i in range(length) if not buying_prices_circularity[i]])])

            demands = [Demand.build_demand_data(data, timestep, self.env_step, max_length) for data, timestep in zip(demands_data, demands_timestep)]
            generations = [Generation.get_generation(data, timestep, self.env_step, max_length) for data, timestep in zip(generations_data, generations_timestep)]
            selling_prices = [SellingPrice.build_selling_price_data(data, timestep, self.env_step, max_length) for data, timestep in zip(selling_prices_data, selling_prices_timestep)]
            buying_prices = [BuyingPrice.build_buying_price_data(data, timestep, self.env_step, max_length, circular) for data, timestep, circular in zip(buying_prices_data, buying_prices_timestep, buying_prices_circularity)]

            return (jax.tree.map(lambda *vals: jnp.array(vals), *demands),
                    jax.tree.map(lambda *vals: jnp.array(vals), *generations),
                    jax.tree.map(lambda *vals: jnp.array(vals), *selling_prices),
                    jax.tree.map(lambda *vals: jnp.array(vals), *buying_prices))


        self.demands_battery_houses, self.generations_battery_houses, self.selling_prices_battery_houses, self.buying_prices_battery_houses = setup_demand_generation_prices(
            settings['demands_battery_houses'],
            settings['generations_battery_houses'],
            settings['selling_prices_battery_houses'],
            settings['buying_prices_battery_houses'],
            self.num_battery_agents)

        self.demands_passive_houses, self.generations_passive_houses, self.selling_prices_passive_houses, self.buying_prices_passive_houses = setup_demand_generation_prices(
            settings['demands_passive_houses'],
            settings['generations_passive_houses'],
            settings['selling_prices_passive_houses'],
            settings['buying_prices_passive_houses'],
            self.num_passive_houses)


        self._termination = settings['termination']

        assert self._termination['max_iterations'] is not None

        max_iterations = self._termination['max_iterations']

        self.trading_coeff = settings['reward']['trading_coeff'] if 'trading_coeff' in settings['reward'] else 0
        self.op_cost_coeff = settings['reward']['operational_cost_coeff'] if 'operational_cost_coeff' in settings['reward'] else 0
        self.deg_coeff = settings['reward']['degradation_coeff'] if 'degradation_coeff' in settings['reward'] else 0
        self.clip_action_coeff = settings['reward']['clip_action_coeff'] if 'clip_action_coeff' in settings['reward'] else 0
        self.use_reward_normalization = settings['use_reward_normalization']

        self.trad_norm_term = max(self.generation_data.max * self.selling_price_data.max,
                                   self.demand_data.max * self.buying_price_data.max)


        ########################## OBSERVATION SPACES ##########################

        self.observation_spaces = OrderedDict([(a, OrderedDict()) for a in self.agents])

        self._obs_battery_agents_keys = ['temperature', 'soc', 'demand', 'generation', 'buying_price', 'selling_price']

        for a in self.battery_agents:
            self.observation_spaces[a]['temperature'] = spaces.Box(low=250., high=400., shape=(1,))
            self.observation_spaces[a]['soc'] = spaces.Box(low=0., high=1., shape=(1,))
            self.observation_spaces[a]['demand'] = spaces.Box(low=0., high=1., shape=(1,))
            self.observation_spaces[a]['generation'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.observation_spaces[a]['buying_price'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))
            self.observation_spaces[a]['selling_price'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))

        # Add optional 'State of Health' in observation space
        if settings['soh']:
            # spaces['soh'] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self._obs_battery_agents_keys.append('soh')
            for a in self.battery_agents:
                self.observation_spaces[a]['soh'] = spaces.Box(low=0., high=1., shape=(1,))

        if settings['day_of_year']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_battery_agents_keys.append('sin_day_of_year')
            self._obs_battery_agents_keys.append('cos_day_of_year')
            for a in self.battery_agents:
                self.observation_spaces[a]['sin_day_of_year'] = spaces.Box(low=-1., high=1., shape=(1,))
                self.observation_spaces[a]['cos_day_of_year'] = spaces.Box(low=-1., high=1., shape=(1,))

        if settings['seconds_of_day']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_battery_agents_keys.append('sin_seconds_of_day')
            self._obs_battery_agents_keys.append('cos_seconds_of_day')
            for a in self.battery_agents:
                self.observation_spaces[a]['sin_seconds_of_day'] = spaces.Box(low=-1., high=1., shape=(1,))
                self.observation_spaces[a]['cos_seconds_of_day'] = spaces.Box(low=-1., high=1., shape=(1,))

        if settings['energy_level']:
            self._obs_battery_agents_keys.append('energy_level')
            for i, a in enumerate(self.battery_agents):
                min_energy = batteries[i].nominal_capacity * batteries[i].soc_state.soc_min * batteries[i].v_max
                max_energy = batteries[i].nominal_capacity * batteries[i].soc_state.soc_max * batteries[i].v_min
                self.observation_spaces[a]['energy_level'] = spaces.Box(low=min_energy, high=max_energy, shape=(1,))


        if settings['network_REC_plus']:
            self._obs_battery_agents_keys.append('network_REC_plus')
            for a in self.battery_agents:
                self.observation_spaces[a]['network_REC_plus'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))

        if settings['network_REC_minus']:
            self._obs_battery_agents_keys.append('network_REC_minus')
            for a in self.battery_agents:
                self.observation_spaces[a]['network_REC_minus'] = spaces.Box(low=0., high=jnp.inf, shape=(1,))

        if settings['network_REC_diff']:
            self._obs_battery_agents_keys.append('network_REC_diff')
            for a in self.battery_agents:
                self.observation_spaces[a]['network_REC_diff'] = spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(1,))

        # if settings['incentive']:
        #     self._obs_battery_agents_keys.append('incentive')
        #     for a in self.battery_agents:
        #         self.observation_spaces[a]['incentive'] = Box(low=0., high=1., shape=(1,))


        self._obs_battery_agents_idx = {key: i for i, key in enumerate(self._obs_battery_agents_keys)}

        self._obs_rec_keys = ['demands_base_battery_houses', 'demands_battery_battery_houses', 'generations_base_battery_houses',
                              'demands_passive_houses', 'generations_passive_houses']

        self.observation_spaces[self.rec_agent]['demands_base_battery_houses'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,))
        self.observation_spaces[self.rec_agent]['demands_battery_battery_houses'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,))
        self.observation_spaces[self.rec_agent]['generations_base_battery_houses'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,))
        self.observation_spaces[self.rec_agent]['demands_passive_houses'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,))
        self.observation_spaces[self.rec_agent]['generations_passive_houses'] = spaces.Box(low=0., high=jnp.inf, shape=(self.num_battery_agents,))
        if 'day_of_year' in settings['REC_obs']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_rec_keys.append('sin_day_of_year')
            self._obs_rec_keys.append('cos_day_of_year')
            self.observation_spaces[self.rec_agent]['sin_day_of_year'] = spaces.Box(low=-1., high=1., shape=(1,))
            self.observation_spaces[self.rec_agent]['cos_day_of_year'] = spaces.Box(low=-1., high=1., shape=(1,))
        if 'seconds_of_day' in settings['REC_obs']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_rec_keys.append('sin_seconds_of_day')
            self._obs_rec_keys.append('cos_seconds_of_day')
            self.observation_spaces[self.rec_agent]['sin_seconds_of_day'] = spaces.Box(low=-1., high=1., shape=(1,))
            self.observation_spaces[self.rec_agent]['cos_seconds_of_day'] = spaces.Box(low=-1., high=1., shape=(1,))


        self.observation_spaces = {key: spaces.Dict(sp) for key, sp in self.observation_spaces.items()}

        self._obs_rec_idx = {key: i for i, key in enumerate(self._obs_rec_keys)}

        self.init_state = EnvState(battery_states=battery_states,
                                   iter=0,
                                   is_rec_turn=False,
                                   timeframe=0,
                                   done=jnp.zeros(shape=(self.num_agents,), dtype=bool),
                                   step=-1)

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

    def _calc_balances_prev_step(self, state: EnvState):
        demands_batteries = self._get_demands(self.demands_battery_houses, state.timeframe-self.env_step)
        generations_batteries = self._get_generations(self.generations_battery_houses, state.timeframe-self.env_step)

        power_batteries = state.battery_states.electrical_state.p       # TODO NON HO IDEA SE SIA CORRETTO CHECK SIGN

        balance_battery_houses = generations_batteries - demands_batteries - power_batteries

        demands_passive_houses = self._get_demands(self.demands_passive_houses, state.timeframe-self.env_step)
        generations_passive_houses = self._get_generations(self.generations_passive_houses, state.timeframe-self.env_step)

        balance_passive_houses = generations_passive_houses - demands_passive_houses

        balances = jnp.concat([balance_battery_houses, balance_passive_houses])

        balance_plus = jnp.where(balances >= 0, balances, 0).sum()
        balance_minus = jnp.where(balances < 0, balances, 0).sum()

        return balance_plus, -balance_minus


    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        demands_batteries = self._get_demands(self.demands_battery_houses, state.timeframe)
        generations_batteries = self._get_generations(self.generations_battery_houses, state.timeframe)
        buying_price_batteries = self._get_buying_prices(self.buying_prices_battery_houses, state.timeframe)
        selling_price_batteries = self._get_selling_prices(self.selling_prices_battery_houses, state.timeframe)

        def batteries_turn():
            temperatures = state.battery_states.thermal_state.temp
            soc = state.battery_states.soc_state.soc

            agg_obs = {'demand': demands_batteries,
                   'generation': generations_batteries,
                   'buying_price': buying_price_batteries,
                   'selling_price': selling_price_batteries,
                   'temperature': temperatures,
                   'soc': soc}

            if 'soh' in self._obs_battery_agents_keys:
                agg_obs['soh'] = state.battery_states.soh
            if 'sin_seconds_of_day' in self._obs_battery_agents_keys:
                agg_obs['sin_seconds_of_day'] = jnp.full(shape=(self.num_battery_agents,), fill_value=jnp.sin(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe))
            if 'cos_seconds_of_day' in self._obs_battery_agents_keys:
                agg_obs['cos_seconds_of_day'] = jnp.full(shape=(self.num_battery_agents,), fill_value=jnp.cos(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe))
            if 'sin_day_of_year' in self._obs_battery_agents_keys:
                agg_obs['sin_day_of_year'] = jnp.full(shape=(self.num_battery_agents,), fill_value=jnp.sin(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe))
            if 'cos_day_of_year' in self._obs_battery_agents_keys:
                agg_obs['cos_day_of_year'] = jnp.full(shape=(self.num_battery_agents,), fill_value=jnp.cos(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe))
            if 'energy_level' in self._obs_battery_agents_keys:
                agg_obs['energy_level'] = state.battery_states.c_max * state.battery_states.electrical_state.v * state.battery_states.soc_state.soc

            balance_plus, balance_minus = self._calc_balances_prev_step(state)

            if 'network_REC_plus' in self._obs_battery_agents_keys:
                agg_obs['network_REC_plus'] = jnp.full(shape=(self.num_battery_agents,), fill_value=balance_plus)
            if 'network_REC_minus' in self._obs_battery_agents_keys:
                agg_obs['network_REC_minus'] = jnp.full(shape=(self.num_battery_agents,), fill_value=balance_minus)
            if 'network_REC_diff' in self._obs_battery_agents_keys:
                agg_obs['network_REC_diff'] = jnp.full(shape=(self.num_battery_agents,), fill_value=balance_plus-balance_minus)

            obs = {a: {key: agg_obs[key][i] for key in self._obs_battery_agents_keys} for i, a in enumerate(self.battery_agents)}


            rec_obs = {'demand_base_battery_houses': jnp.zeros(self.num_battery_agents),
                       'demands_battery_battery_houses': jnp.zeros(self.num_battery_agents),
                       'generations_battery_houses': jnp.zeros(self.num_battery_agents),
                       'demands_passive_houses': jnp.zeros(self.num_passive_houses),
                       'generations_passive_houses': jnp.zeros(self.num_passive_houses)}

            for o in ['sin_seconds_of_day', 'cos_seconds_of_day', 'sin_day_of_year', 'cos_day_of_year']:
                if o in self._obs_rec_keys:
                    rec_obs[o] = 0.

            obs[self.rec_agent] = rec_obs

            return obs

        def rec_turn():
            obs_battery_agents = {'demand': 0.,
                       'generation': 0.,
                       'buying_price': 0.,
                       'selling_price': 0.,
                       'temperature': 0.,
                       'soc': 0.}

            for o in ['soh', 'sin_seconds_of_day', 'cos_seconds_of_day', 'sin_day_of_year', 'cos_day_of_year', 'energy_level', 'network_REC_plus', 'network_REC_minus', 'network_REC_diff']:
                obs_battery_agents[o] = 0.

            obs = {a: obs_battery_agents for a in self.battery_agents}

            demands_passive_houses = self._get_demands(self.demands_passive_houses, state.timeframe)
            generations_passive_houses = self._get_generations(self.generations_passive_houses, state.timeframe)

            rec_obs = {'demand_base_battery_houses': demands_batteries,
                       'demands_battery_battery_houses': state.battery_states.electrical_state.p,                   # TODO NON HO IDEA SE SIA CORRETTO CHECK SIGN
                       'generations_battery_houses': generations_batteries,
                       'demands_passive_houses': demands_passive_houses,
                       'generations_passive_houses': generations_passive_houses}

            if 'sin_seconds_of_day' in self._obs_rec_keys:
                rec_obs['sin_seconds_of_day'] = jnp.sin(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe)
            if 'cos_seconds_of_day' in self._obs_rec_keys:
                rec_obs['cos_seconds_of_day'] = jnp.cos(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe)
            if 'sin_day_of_year' in self._obs_rec_keys:
                rec_obs['sin_day_of_year'] = jnp.sin(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe)
            if 'cos_day_of_year' in self._obs_rec_keys:
                rec_obs['cos_day_of_year'] = jnp.cos(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe)

            obs[self.rec_agent] = rec_obs

            return obs

        return jax.lax.cond(state.is_rec_turn, rec_turn, batteries_turn)

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        return self.get_obs(self.initial_state), self.init_state

    def step_env(self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        return jax.lax.cond(state.is_rec_turn,
                            self.step_rec,
                            self.step_batteries,
                            state, actions)

    def step_rec(self, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        ...

    def step_batteries(self, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array([actions[a] for a in self.num_battery_agents])

        new_timeframe = state.timeframe + self.env_step
        last_v = state.battery_states.electrical_state.v
        i_max, i_min = jax.vmap(self.BESS_class.get_feasible_current, in_axes=(0, 0, None))(state.battery_states, state.battery_states.soc_state.soc, self.env_step)

        i_to_apply = jnp.clip(actions, i_min, i_max)

        to_load = last_v * i_to_apply

        demands = self._get_demands(self.demands_battery_houses, state.timeframe)
        generations = self._get_generations(self.generations_battery_houses, state.timeframe)

        to_trade = generations - demands - to_load

        old_soh = state.battery_states.soh

        new_battery_states = jax.vmap(self.BESS_class.step, in_axes=(0, 0, None))(state.battery_states, i_to_apply, self.env_step)

        buying_prices = self._get_buying_prices(self.buying_prices_battery_houses, state.timeframe)
        selling_prices = self._get_selling_prices(self.selling_prices_battery_houses, state.timeframe)

        r_trading = jnp.minimum(0, to_trade) * buying_prices + jnp.maximum(0, to_trade) * selling_prices

        r_clipping = jnp.abs((actions - i_to_apply) * last_v)

        r_deg = self._calc_deg_reward(old_soh, new_battery_states.soh, new_battery_states.nominal_cost, self._termination['min_soh'])

        r_op = self._calc_op_reward(new_battery_states.nominal_cost,
                                    new_battery_states.nominal_capacity * new_battery_states.nominal_voltage / 1000,
                                    new_battery_states.c_max * new_battery_states.nominal_voltage / 1000,
                                    new_battery_states.nominal_dod,
                                    new_battery_states.nominal_lifetime,
                                    new_battery_states.nominal_voltage,
                                    new_battery_states.electrical_state.p,
                                    new_battery_states.electrical_state.r0,
                                    new_battery_states.electrical_state.rc.resistance,
                                    new_battery_states.soc_state.soc,
                                    new_battery_states.electrical_state.p <= 0)

        norm_r_trading, norm_r_op, norm_r_deg, norm_r_clipping = self._normalize_reward(new_battery_states, r_trading, r_op, r_deg, r_clipping)
        weig_r_trading, weig_r_op, weig_r_deg, weig_r_clipping = (self.trading_coeff * norm_r_trading, self.op_cost_coeff * norm_r_op,
                                                                  self.deg_coeff * norm_r_deg, self.clip_action_coeff * norm_r_clipping)

        r_tot = weig_r_trading + weig_r_op + weig_r_deg + weig_r_clipping

        new_iteration = state.iter + 1

        terminated = new_battery_states.soh <= self._termination['min_soh']

        truncated = jnp.logical_or(new_iteration >= self._termination['max_iterations'],
                                   jnp.logical_or(jnp.logical_or(jax.vmap(Demand.is_run_out_of_data, in_axes=(0, None))(self.demands_battery_houses, new_timeframe),
                                                                 jax.vmap(Generation.is_run_out_of_data, in_axes=(0, None))(self.generations_battery_houses, new_timeframe)),
                                                  jnp.logical_or(jax.vmap(BuyingPrice.is_run_out_of_data, in_axes=(0, None))(self.buying_prices_battery_houses, new_timeframe),
                                                                 jax.vmap(SellingPrice.is_run_out_of_data, in_axes=(0, None))(self.selling_prices_battery_houses, new_timeframe))),
                                   )

        new_state = state.replace(battery_states=new_battery_states,
                                  iter=new_iteration,
                                  timeframe=new_timeframe,
                                  is_rec_turn=True)

        rewards = {a: r_tot[i] for i, a in enumerate(self.battery_agents)}
        rewards[self.rec_agent] = 0.

        packed_info = {'soc': new_battery_states.soc_state.soc,
                       'soh': new_battery_states.soh,
                       'pure_reward': {'r_trad': r_trading,
                                       'r_op': r_op,
                                       'r_deg': r_deg,
                                       'r_clipping': r_clipping},
                       'norm_reward': {'r_trad': norm_r_trading,
                                       'r_op': norm_r_op,
                                       'r_deg': norm_r_deg,
                                       'r_clipping': norm_r_clipping},
                       'weig_reward': {'r_trad': weig_r_trading,
                                       'r_op': weig_r_op,
                                       'r_deg': weig_r_deg,
                                       'r_clipping': weig_r_clipping},
                       'r_tot': r_tot,
                       'r_glob': jnp.zeros(self.num_battery_agents)}

        info = {a: jax.tree.map(lambda val: val[i], packed_info) for i, a in enumerate(self.battery_agents)}
        dones_array = jnp.logical_or(truncated, terminated)

        dones = {a: dones_array[i] for i, a in enumerate(self.battery_agents)}
        dones[self.rec_agent] = False
        dones['__all__'] = jnp.all(dones_array)

        info[self.rec_agent] = {'self_consumption': 0.,
                                'tot_incentives': 0.,
                                'reward': 0.}

        return self.get_obs(new_state), new_state, rewards, dones, info


    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def _calc_deg_reward(self, old_soh, curr_soh, replacement_cost, soh_limit):

        delta_soh = jnp.abs(old_soh - curr_soh)
        soh_cost = delta_soh * replacement_cost / (1 - soh_limit)
        return - soh_cost

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

    def _normalize_reward(self, battery_states: BessState, r_trading, r_op, r_deg, r_clipping):
        norm_r_trading = r_trading / self.trad_norm_term
        norm_r_op = r_op / battery_states.nominal_cost
        norm_r_clipping = r_clipping / jnp.maximum(jnp.abs(self.demands_battery_houses.max - self.demands_battery_houses.min),
                                                   jnp.abs(self.generations_battery_houses.max - self.generations_battery_houses.min))

        return norm_r_trading, norm_r_op, r_deg, norm_r_clipping