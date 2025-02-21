import chex
from flax import struct
import jax
import jax.numpy as jnp
# import numpy as np

from gymnax.environments.environment import TEnvParams, TEnvState

from ernestogym.envs_jax.single_agent.env import MicroGridEnv, EnvState, EnvParams


from ernestogym.ernesto_jax.demand import Demand, DemandData
from ernestogym.ernesto_jax.generation import Generation
from ernestogym.ernesto_jax.market import BuyingPrice, SellingPrice
from ernestogym.ernesto_jax.ambient_temperature import AmbientTemperature

from typing import Dict, Union, Tuple, Any


class MicroGridEnvSocAction(MicroGridEnv):

    def __init__(self, settings: Dict, battery_type: str, demand_profile: str = None):
        super().__init__(settings, battery_type, demand_profile)

    def step_env(self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array], params: EnvParams) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        action = jnp.array(action).flatten()[0] #FIXME dava problemi

        new_timeframe = state.timeframe + params.env_step
        last_v = state.battery_state.electrical_state.v
        i_max, i_min = self.BESS.get_feasible_current(state.battery_state, state.battery_state.soc_state.soc, params.env_step)

        i = (action - state.battery_state.soc_state.soc) * state.battery_state.c_max * 3600 / params.env_step

        i_to_apply = jnp.clip(i, i_min, i_max)

        old_soh = state.battery_state.soh

        t_amb = AmbientTemperature.get_amb_temperature(self.temp_amb, new_timeframe)

        new_battery_state = self.BESS.step(state.battery_state, i_to_apply, dt=params.env_step, t_amb=t_amb)

        to_load = new_battery_state.electrical_state.p

        to_trade = Generation.get_generation(self.generation_data, new_timeframe) - Demand.get_demand(state.demand_data, new_timeframe) - to_load

        r_trading = jnp.minimum(0, to_trade) * BuyingPrice.get_buying_price(self.buying_price_data, new_timeframe) + jnp.maximum(0, to_trade) * SellingPrice.get_selling_price(self.selling_price_data, new_timeframe)

        r_clipping = 0.     #-jnp.abs((action - i_to_apply) * last_v)

        r_deg = self._calc_deg_reward(old_soh, new_battery_state.soh, new_battery_state.nominal_cost, self._termination['min_soh'])
        # r_op = self._calc_op_reward(replacement_cost=new_battery_state.nominal_cost,
        #                             C_rated=new_battery_state.nominal_capacity * new_battery_state.nominal_voltage / 1000,
        #                             C=new_battery_state.c_max * new_battery_state.nominal_voltage / 1000,
        #                             DoD_rated=new_battery_state.nominal_dod,
        #                             L_rated=new_battery_state.nominal_lifetime,
        #                             v_rated=new_battery_state.nominal_voltage,
        #                             K_rated=new_battery_state.electrical_state.rc.resistance,
        #                             p=new_battery_state.electrical_state.p,
        #                             r=new_battery_state.electrical_state.r0,
        #                             soc=new_battery_state.soc_state.soc,
        #                             is_discharging=(new_battery_state.electrical_state.p <= 0))

        r_deg = 0.
        r_op = 0.

        norm_r_trading, norm_r_op, norm_r_deg, norm_r_clipping = self._normalize_reward(state, new_battery_state, r_trading, r_op, r_deg, r_clipping, params)
        weig_r_trading, weig_r_op, weig_r_deg, weig_r_clipping = (params.trading_coeff * norm_r_trading, params.op_cost_coeff * norm_r_op,
                                                                  params.deg_coeff * norm_r_deg, params.clip_action_coeff * norm_r_clipping)

        r_tot = weig_r_trading + weig_r_op + weig_r_deg + weig_r_clipping

        new_time = state.time + params.env_step
        new_iteration = state.iteration + 1

        terminated = new_battery_state.soh <= self._termination['min_soh']

        truncated = jnp.logical_or(new_iteration >= self._termination['max_iterations'],
                                   jnp.logical_or(jnp.logical_or(Demand.is_run_out_of_data(state.demand_data, new_timeframe),
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
                'r_tot': r_tot,
                'i_to_apply': i_to_apply}       #FIXME REMOVE

        return self.get_obs(new_state, params), new_state, r_tot, jnp.logical_or(terminated, truncated), info