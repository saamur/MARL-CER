import chex
from flax import struct
import jax
import jax.numpy as jnp

from gymnax.environments import environment
from gymnax.environments import spaces
from gymnax.environments.environment import TEnvParams, TEnvState

from ernestogym.ernesto_jax.energy_storage.bess import BessState

import ernestogym.ernesto_jax.energy_storage.bess_fading as bess_fading
import ernestogym.ernesto_jax.energy_storage.bess_degrading as bess_degrading
import ernestogym.ernesto_jax.energy_storage.bess_degrading_dropflow as bess_degrading_dropflow

from ernestogym.ernesto_jax.demand import Demand
from ernestogym.ernesto_jax.generation import Generation
from ernestogym.ernesto_jax.market import BuyingPrice, SellingPrice

from typing import Dict, Union, Tuple, Any
from collections import OrderedDict

@struct.dataclass
class EnvState(environment.EnvState):
    battery_state: BessState

    iteration: int = 0.
    timeframe: int = 0

@struct.dataclass
class EnvParams(environment.EnvParams):
    trading_coeff: float = 0
    op_cost_coeff: float = 0
    deg_coeff: float = 0
    clip_action_coeff: float = 0
    use_reward_normalization: bool = False
    trad_norm_term: float = 1.

    i_min_action: float = -1
    i_max_action: float = 1
    env_step: int = 60


class MicroGridEnv(environment.Environment[EnvState, EnvParams]):
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = 60 * 60 * 24
    DAYS_PER_YEAR = 365.25

    def __init__(self, settings: Dict, battery_type: str):

        if battery_type == 'fading':
            self.BESS = bess_fading.BatteryEnergyStorageSystem
            battery_state = bess_fading.BatteryEnergyStorageSystem.get_init_state(models_config=settings['models_config'],
                                                                                  battery_options=settings['battery'],
                                                                                  input_var=settings['input_var'])
        elif battery_type == 'degrading':
            self.BESS = bess_degrading.BatteryEnergyStorageSystem
            battery_state = bess_degrading.BatteryEnergyStorageSystem.get_init_state(models_config=settings['models_config'],
                                                                                     battery_options=settings['battery'],
                                                                                     input_var=settings['input_var'])
        elif battery_type == 'degrading_dropflow':
            self.BESS = bess_degrading_dropflow.BatteryEnergyStorageSystem
            battery_state = bess_degrading_dropflow.BatteryEnergyStorageSystem.get_init_state(models_config=settings['models_config'],
                                                                                              battery_options=settings['battery'],
                                                                                              input_var=settings['input_var'])
        else:
            raise ValueError(f'Unsupported battery type: {battery_type}')

        # reset_params ?
        # params_bounds ?

        env_step = settings['step']

        self.demand_data = Demand.build_demand_data(jnp.array(settings['demand']['data']['64']), in_timestep=self.SECONDS_PER_MINUTE, out_timestep=env_step)
        self.generation_data = Generation.build_generation_data(jnp.array(settings['generation']['data']['PV']), in_timestep=self.SECONDS_PER_MINUTE, out_timestep=env_step)
        self.buying_price_data = BuyingPrice.build_buying_price_data(jnp.array(settings['market']['data']['ask']), in_timestep=self.SECONDS_PER_HOUR, out_timestep=env_step)
        self.selling_price_data = SellingPrice.build_selling_price_data(jnp.array(settings['market']['data']['bid']), in_timestep=self.SECONDS_PER_HOUR, out_timestep=env_step)




        self._termination = settings['termination']

        assert self._termination['max_iterations'] is not None

        max_iterations = self._termination['max_iterations']

        trading_coeff = settings['reward']['trading_coeff'] if 'trading_coeff' in settings['reward'] else 0
        op_cost_coeff = settings['reward']['operational_cost_coeff'] if 'operational_cost_coeff' in settings['reward'] else 0
        deg_coeff = settings['reward']['degradation_coeff'] if 'degradation_coeff' in settings['reward'] else 0
        clip_action_coeff = settings['reward']['clip_action_coeff'] if 'clip_action_coeff' in settings['reward'] else 0
        use_reward_normalization = settings['use_reward_normalization']

        trad_norm_term = max(self.generation_data.max * self.selling_price_data.max,
                             self.demand_data.max * self.buying_price_data.max)

        # eval_profile ?

        self.spaces = OrderedDict()

        self.spaces['temperature'] = {'low': 250., 'high': 400.}
        self.spaces['soc'] = {'low': 0., 'high': 1.}
        self.spaces['demand'] = {'low': 0., 'high': jnp.inf}
        self._obs_keys = ['temperature', 'soc', 'demand']

        # Add optional 'State of Health' in observation space
        if settings['soh']:
            # spaces['soh'] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self._obs_keys.append('soh')
            self.spaces['soh'] = {'low': 0., 'high': 1.}

            # Add optional 'generation' in observation space
        if self.generation_data is not None:
            # spaces['generation'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)s
            self._obs_keys.append('generation')
            self.spaces['generation'] = {'low': 0., 'high': jnp.inf}

        # Add optional 'bid' and 'ask' of energy market in observation space
        if self.buying_price_data is not None:
            # spaces['buying_price'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            self._obs_keys.append('buying_price')
            self.spaces['buying_price'] = {'low': 0., 'high': jnp.inf}

        if self.selling_price_data is not None:
            # spaces['selling_price'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            self._obs_keys.append('selling_price')
            self.spaces['selling_price'] = {'low': 0., 'high': jnp.inf}

        if settings['day_of_year']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_keys.append('sin_day_of_year')
            self._obs_keys.append('cos_day_of_year')
            self.spaces['sin_day_of_year'] = {'low': -1, 'high': 1}
            self.spaces['cos_day_of_year'] = {'low': -1, 'high': 1}

        if settings['seconds_of_day']:
            # spaces['seconds_of_day'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_keys.append('sin_seconds_of_day')
            self._obs_keys.append('cos_seconds_of_day')
            self.spaces['sin_seconds_of_day'] = {'low': -1, 'high': 1}
            self.spaces['cos_seconds_of_day'] = {'low': -1, 'high': 1}

        if settings['energy_level']:
            self._obs_keys.append('energy_level')
            min_energy = battery_state.nominal_capacity * battery_state.soc_state.soc_min * battery_state.v_max
            max_energy = battery_state.nominal_capacity * battery_state.soc_state.soc_max * battery_state.v_min
            self.spaces['energy_level'] = {'low': min_energy, 'high': max_energy}

        self._obs_idx = {key: i for i, key in enumerate(self._obs_keys)}

        i_max_action = self.BESS.get_feasible_current(battery_state, battery_state.soc_state.soc_min, dt=env_step)[0]
        i_min_action = self.BESS.get_feasible_current(battery_state, battery_state.soc_state.soc_max, dt=env_step)[1]

        self.initial_state = EnvState(iteration=0,
                                      time=0,
                                      battery_state=battery_state)

        self.initial_state = jax.tree.map(lambda leaf: jnp.array(leaf), self.initial_state)

        self.params = EnvParams(trading_coeff=trading_coeff,
                                op_cost_coeff=op_cost_coeff,
                                deg_coeff=deg_coeff,
                                clip_action_coeff=clip_action_coeff,
                                use_reward_normalization=use_reward_normalization,
                                trad_norm_term=trad_norm_term,
                                i_min_action=i_min_action,
                                i_max_action=i_max_action,
                                env_step=env_step)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def observation_space(self, params: EnvParams):
        return spaces.Dict({key: spaces.Box(self.spaces[key]['low'], self.spaces[key]['high'], shape=(1,)) for key in self._obs_keys})

    def state_space(self, params: EnvParams):
        return self.observation_space(params)

    def action_space(self, params: EnvParams):
        return spaces.Box(params.i_min_action, params.i_max_action, shape=(1,))

    def get_obs(self, state: EnvState, params=None, key=None):
        obs = jnp.empty(len(self._obs_keys))

        for i, key in enumerate(self._obs_keys):
            match key:
                case 'temperature':
                    val = state.battery_state.thermal_state.temp

                case 'soc':
                    val = state.battery_state.soc_state.soc

                case 'demand':
                    val = Demand.get_demand(self.demand_data, state.time)

                case 'aging':
                    val = state.battery_state.soh

                case 'generation':
                    val = Generation.get_generation(self.generation_data, state.time)

                case 'buying_price':
                    val = BuyingPrice.get_buying_price(self.buying_price_data, state.time)

                case 'selling_price':
                    val = SellingPrice.get_selling_price(self.selling_price_data, state.time)

                case 'sin_day_of_year':
                    sin_year = jnp.sin(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe)
                    val = sin_year

                case 'cos_day_of_year':
                    cos_year = jnp.cos(2 * jnp.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * state.timeframe)
                    val = cos_year

                case 'sin_seconds_of_day':
                    sin_day = jnp.sin(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe)
                    val = sin_day

                case 'cos_seconds_of_day':
                    cos_day = jnp.cos(2 * jnp.pi / self.SECONDS_PER_DAY * state.timeframe)
                    val = cos_day

                case 'energy_level':
                    val = state.battery_state.c_max * state.battery_state.electrical_state.v * state.battery_state.soc_state.soc

            obs = obs.at[i].set(val)

        return obs

    def reset_env(self, key: chex.PRNGKey, params: TEnvParams):
        state = self.initial_state
        obs = self.get_obs(state, params)
        return obs, state

    def step_env(self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array], params: EnvParams) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:

        obs_pre_step = self.get_obs(state, params)

        new_timeframe = state.timeframe + params.env_step
        last_v = state.battery_state.electrical_state.v
        i_max, i_min = self.BESS.get_feasible_current(state.battery_state, state.battery_state.soc_state.soc, params.env_step)

        i_to_apply = jnp.clip(action, i_min, i_max)

        to_load = last_v * i_to_apply

        to_trade = Demand.get_demand(self.demand_data, new_timeframe) - Generation.get_generation(self.generation_data, new_timeframe) - to_load

        old_soh = state.battery_state.soh

        new_battery_state = self.BESS.step(state.battery_state, i_to_apply, dt=params.env_step)

        r_trading = jnp.minimum(0, to_trade) * obs_pre_step[self._obs_idx['buying_price']] + jnp.maximum(0, to_trade) * obs_pre_step[self._obs_idx['selling_price']]

        r_clipping = jnp.abs((action - i_to_apply) * last_v)

        r_deg = self._calc_deg_reward(old_soh, new_battery_state.soh, new_battery_state.nominal_cost, self._termination['min_soh'])
        r_op = self._calc_op_reward(replacement_cost=new_battery_state.nominal_cost,
                                    C_rated=new_battery_state.nominal_capacity * new_battery_state.nominal_voltage / 1000,
                                    C=new_battery_state.c_max * new_battery_state.nominal_voltage / 1000,
                                    DoD_rated=new_battery_state.nominal_dod,
                                    L_rated=new_battery_state.nominal_lifetime,
                                    v_rated=new_battery_state.nominal_voltage,
                                    K_rated=new_battery_state.electrical_state.rc.resistance,
                                    p=new_battery_state.electrical_state.p,
                                    r=new_battery_state.electrical_state.r0,
                                    soc=new_battery_state.soc_state.soc,
                                    is_discharging=(new_battery_state.electrical_state.p <= 0))

        norm_r_trading, norm_r_op, norm_r_deg, norm_r_clipping = self._normalize_reward(new_battery_state, r_trading, r_op, r_deg, r_clipping, params)
        weig_r_trading, weig_r_op, weig_r_deg, weig_r_clipping = (params.trading_coeff * norm_r_trading, params.op_cost_coeff * norm_r_op,
                                                                  params.deg_coeff * norm_r_deg, params.clip_action_coeff * norm_r_clipping)

        r_tot = weig_r_trading + weig_r_op + weig_r_deg + weig_r_clipping

        new_time = state.time + params.env_step
        new_iteration = state.iteration + 1

        terminated = new_battery_state.soh <= self._termination['min_soh']

        truncated = jnp.logical_or(new_iteration >= self._termination['max_iterations'],
                                   jnp.logical_or(jnp.logical_or(Demand.is_run_out_of_data(self.demand_data, new_timeframe),
                                                                 Generation.is_run_out_of_data(self.generation_data, new_timeframe)),
                                                  jnp.logical_or(BuyingPrice.is_run_out_of_data(self.buying_price_data, new_timeframe),
                                                                 SellingPrice.is_run_out_of_data(self.selling_price_data, new_timeframe))),
                                   )

        new_state = state.replace(battery_state=new_battery_state,
                                  iteration=new_iteration,
                                  timeframe=new_timeframe,
                                  time=new_time
        )

        info = {'soc': new_battery_state.soc_state.soc,
                'soh': new_battery_state.soh,
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
                'r_tot': r_tot}

        return self.get_obs(new_state, params), new_state, r_tot, jnp.logical_or(terminated, truncated), info

    def _calc_deg_reward(self, old_soh, curr_soh, replacement_cost, soh_limit):

        delta_soh = jnp.abs(old_soh - curr_soh)
        soh_cost = delta_soh * replacement_cost / (1 - soh_limit)
        return - soh_cost

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
        soc = jax.lax.cond(soc == 0,
                           lambda : 1e-6,
                           lambda : soc)

        # Coefficient c_avai = c_bat
        c_bat = replacement_cost / (C_rated * DoD_rated * (0.9 * L_rated - 0.1))

        # P_loss depending on P charged or discharged
        h_bat = jax.lax.cond(is_discharging,
                             lambda : jnp.abs(p) + (1 * (r + K_rated / soc) / v_rated ** 2 * p ** 2 +
                                                    1 * C * K_rated * (1 - soc) / (soc * v_rated ** 2) * p),
                             lambda : (1 * (r + K_rated / (0.9 - soc)) / v_rated ** 2 * p ** 2 +
                                       1 * C * K_rated * (1 - soc) / (soc * v_rated ** 2) * p))

        # h_bat = jax.lax.select(is_discharging,
        #                      jnp.abs(p) + (1 * (r + K_rated / soc) / v_rated ** 2 * p ** 2 +
        #                                    1 * C * K_rated * (1 - soc) / (soc * v_rated ** 2) * p),
        #                      (1 * (r + K_rated / (0.9 - soc)) / v_rated ** 2 * p ** 2 +
        #                       1 * C * K_rated * (1 - soc) / (soc * v_rated ** 2) * p))

        # Dividing by 1e3 to convert because it is in €/kWh, to get the cost in €/Wh
        op_cost_term = c_bat * h_bat / 1e3

        return - op_cost_term

    def _normalize_reward(self, battery_state: BessState, r_trading, r_op, r_deg, r_clipping, params: EnvParams):
        norm_r_trading = r_trading / params.trad_norm_term
        norm_r_op = r_op / battery_state.nominal_cost
        norm_r_clipping = r_clipping / jnp.maximum(jnp.abs(self.demand_data.max - self.generation_data.min),
                                                   jnp.abs(self.generation_data.max - self.demand_data.min))

        return norm_r_trading, norm_r_op, r_deg, norm_r_clipping