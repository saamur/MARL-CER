from typing import Dict

from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

from .dropflow import Dropflow, DropflowState

@struct.dataclass
class TimeStressModel:
    k_t: float

@struct.dataclass
class SocStressModel:
    k_soc: float
    soc_ref: float


@struct.dataclass
class TempStressModel:
    k_temp: float
    temp_ref: float

@struct.dataclass
class DodBolunStressModel:
    k_delta1: float
    k_delta2: float
    k_delta3: float


@struct.dataclass
class BolunDropflowState:

    init_soh:float
    soh: float
    soc_mean: float
    temp_battery_mean: float
    cum_sum_temp_history: jnp.ndarray
    cum_sum_soc_history: jnp.ndarray
    n_steps: int

    dropflow_state: DropflowState

    f_cyc: float
    deg: float

    time_stress_model: TimeStressModel
    soc_stress_model: SocStressModel
    temp_stress_model: TempStressModel
    dod_bolun_stress_model: DodBolunStressModel

    alpha_sei:float
    beta_sei:float



class BolunDropflowModel:

    @classmethod
    # @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls, components_setting: Dict, stress_models: Dict, max_length_history=50000, max_length_reversals=10000):

        dropflow_state = Dropflow.get_init_state(max_length_reversals)

        time_stress_model = TimeStressModel(k_t=stress_models['time']['k_t'])
        soc_stress_model = SocStressModel(k_soc=stress_models['soc']['k_soc'],
                                          soc_ref=stress_models['soc']['soc_ref'])
        temp_stress_model = TempStressModel(k_temp=stress_models['temperature']['k_temp'],
                                            temp_ref=stress_models['temperature']['temp_ref'])
        dod_bolun_stress_model = DodBolunStressModel(k_delta1=stress_models['dod_bolun']['k_delta1'],
                                                     k_delta2=stress_models['dod_bolun']['k_delta2'],
                                                     k_delta3=stress_models['dod_bolun']['k_delta3'])

        return BolunDropflowState(init_soh=1.,
                                  soh=1.,
                                  soc_mean=1.,
                                  temp_battery_mean=0.,
                                  cum_sum_temp_history=jnp.zeros(max_length_history+1),
                                  cum_sum_soc_history=jnp.zeros(max_length_history + 1),
                                  n_steps=0,
                                  dropflow_state=dropflow_state,
                                  f_cyc=0.,
                                  deg=0.,
                                  time_stress_model=time_stress_model,
                                  soc_stress_model=soc_stress_model,
                                  temp_stress_model=temp_stress_model,
                                  dod_bolun_stress_model=dod_bolun_stress_model,
                                  alpha_sei=components_setting['SEI']['alpha_sei'],
                                  beta_sei=components_setting['SEI']['beta_sei'])

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_soh(cls, state: BolunDropflowState, temp_battery, temp_ambient, soc, elapsed_time, do_check:bool):     #TODO t_ref è temperatura ambiente o un'altra roba?

        new_dropflow_state = Dropflow.add_point(state.dropflow_state, soc)

        new_cum_sum_temp_history = state.cum_sum_temp_history.at[state.n_steps+1].set(state.cum_sum_temp_history[state.n_steps] + temp_battery)
        new_cum_sum_soc_history = state.cum_sum_soc_history.at[state.n_steps + 1].set(state.cum_sum_soc_history[state.n_steps] + soc)
        new_n_steps = state.n_steps + 1

        new_temp_battery_mean = state.temp_battery_mean + (temp_battery - state.temp_battery_mean) / new_n_steps
        new_soc_mean = state.soc_mean + (soc - state.soc_mean) / new_n_steps

        new_state = state.replace(dropflow_state=new_dropflow_state,
                                  cum_sum_temp_history=new_cum_sum_temp_history,
                                  cum_sum_soc_history=new_cum_sum_soc_history,
                                  n_steps=new_n_steps,
                                  temp_battery_mean=new_temp_battery_mean,
                                  soc_mean=new_soc_mean)

        def check_new_degradation(state: BolunDropflowState):
            #calendar aging
            # jax.debug.print('JAX k_temp: {k_temp}, mean_temp: {mean_temp}, temp_ref: {temp_ref}', k_temp=state.temp_stress_model.k_temp,
            #                 mean_temp=state.temp_battery_mean, temp_ref=state.temp_stress_model.temp_ref, ordered=True)
            #
            # jax.debug.print('JAX k_soc: {k_soc}, soc: {soc}, soc_ref: {soc_ref}', k_soc=state.soc_stress_model.k_soc,
            #                 soc=state.soc_mean, soc_ref=state.soc_stress_model.soc_ref, ordered=True)
            #
            # jax.debug.print('JAX k_t: {k_t}, t: {t}', k_t=state.time_stress_model.k_t,
            #                 t=elapsed_time, ordered=True)

            # soc_mean = state.cum_sum_soc_history[state.n_steps] / state.n_steps
            # temp_battery_mean = state.cum_sum_temp_history[state.n_steps] / state.n_steps

            f_cal = (temperature_stress(state.temp_stress_model.k_temp, state.temp_battery_mean, state.temp_stress_model.temp_ref) *
                     soc_stress(state.soc_stress_model.k_soc, state.soc_mean, state.soc_stress_model.soc_ref) *  # fixme siamo sicuri che devo usare la media del soc?
                     time_stress(state.time_stress_model.k_t, elapsed_time))

            # jax.debug.print('jax f_cal: {x}', x=f_cal, ordered=True)

            new_dropflow_state, rngs, soc_means, counts, i_start, i_end, num_complete_cyc, num_prov_cyc = Dropflow.extract_new_cycles(state.dropflow_state)

            # length = len(state.temp_history)
            # indexes = jnp.arange(length)
            # mask = jnp.logical_and(indexes[:, None] >= i_start[None, :], indexes[:, None] <= i_end[None, :])
            # temp_means = jnp.sum(jnp.where(mask, state.temp_history[:, None], 0), axis=0)

            temp_means = (state.cum_sum_temp_history[i_end] - state.cum_sum_temp_history[i_start]) / (i_end - i_start)
            soc_means = (state.cum_sum_soc_history[i_end] - state.cum_sum_soc_history[i_start]) / (i_end - i_start)

            new_iter_complete_f_cyc, incomplete_f_cyc = cls.compute_cyclic_aging(state,
                                                                                 temp_ambient=temp_ambient,
                                                                                 cycle_type=counts,
                                                                                 cycle_dod=rngs,
                                                                                 avg_cycle_soc=soc_means,
                                                                                 avg_cycle_temp=temp_means,
                                                                                 num_complete_cyc=num_complete_cyc,
                                                                                 num_prov_cyc=num_prov_cyc)

            # jax.debug.print('jax incomplete_f_cyc: {x}', x=incomplete_f_cyc, ordered=True)
            # jax.debug.print('jax new_complete_f_cyc: {x}', x=new_iter_complete_f_cyc, ordered=True)
            #
            f_d = f_cal + state.f_cyc + new_iter_complete_f_cyc + incomplete_f_cyc
            # jax.debug.print('jax prev_complete: {x}', x=state.f_cyc, ordered=True)
            # jax.debug.print('jax f_d: {x}', x=f_d, ordered=True)

            deg = jnp.clip(1 - state.alpha_sei * jnp.exp(-state.beta_sei * f_d) - (1 - state.alpha_sei) * jnp.exp(-f_d), 0, 1)

            new_state = state.replace(n_steps=new_n_steps,
                                      temp_battery_mean=new_temp_battery_mean,
                                      soc_mean=new_soc_mean,
                                      dropflow_state=new_dropflow_state,
                                      deg=deg,
                                      f_cyc=state.f_cyc+new_iter_complete_f_cyc)

            return new_state, deg

        new_state, deg = jax.lax.cond(do_check,
                                      check_new_degradation,
                                      lambda state: (state, state.deg),
                                      new_state)

        new_soh = new_state.init_soh - deg

        new_state = new_state.replace(soh=new_soh)

        return new_state, new_soh


    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_cyclic_aging(cls, state: BolunDropflowState, temp_ambient, cycle_type, cycle_dod, avg_cycle_temp, avg_cycle_soc, num_complete_cyc, num_prov_cyc):
        cyclic_aging = (cycle_type *
                        dod_bolun_stress(cycle_dod, state.dod_bolun_stress_model.k_delta1, state.dod_bolun_stress_model.k_delta2, state.dod_bolun_stress_model.k_delta3) *
                        temperature_stress(state.temp_stress_model.k_temp, avg_cycle_temp, state.temp_stress_model.temp_ref) *
                        soc_stress(state.soc_stress_model.k_soc, avg_cycle_soc, state.soc_stress_model.soc_ref))

        arange = jnp.arange(len(cycle_type))

        new_complete_f_cyc = jnp.sum(jnp.where(arange < num_complete_cyc, cyclic_aging, 0))
        incomplete_f_cyc = jnp.sum(jnp.where(jnp.logical_and(arange >= num_complete_cyc, arange < num_complete_cyc+num_prov_cyc), cyclic_aging, 0))

        return new_complete_f_cyc, incomplete_f_cyc


def temperature_stress(k_temp, mean_temp, temp_ref):
    """
    Stress function caused by the temperature, to be computed in Kelvin.
    This k_tempcan be used only for batteries operating above 15°C.

    Inputs:
    :param k_temp: temperature stress coefficient
    :param mean_temp: current battery temperature
    :param temp_ref: ambient temperature
    """
    return jnp.exp(jnp.float32(k_temp) * (mean_temp - temp_ref) * (temp_ref / mean_temp))

def soc_stress(k_soc, soc, soc_ref):
    """
    Stress function caused by the SoC (State of Charge).

    Inputs:
    :param k_soc: soc stress coefficient
    :param soc: current battery soc
    :param soc_ref: reference soc level, usually around 0.4 to 0.5
    """
    return jnp.exp(k_soc * (soc - soc_ref))

def time_stress(k_t, t):
    """
    Stress function of calendar elapsed time

    Inputs:
    :param k_t: time stress coefficient
    :param t: current battery age
    """
    return jnp.float32(k_t) * t


def dod_bolun_stress(dod, k_delta1, k_delta2, k_delta3):
    """
    Stress function caused by DoD (Depth of Discharge) presented in Bolun's paper.
    This is more accurate to fit with LMO batteries.

    Inputs:
    :param dod: current depth of discharge
    :param k_delta1: first dod stress coefficient
    :param k_delta2: second dod stress coefficient
    :param k_delta3: third dod stress coefficient
    """
    return (jnp.float32(k_delta1) * dod ** jnp.float32(k_delta2) + jnp.float32(k_delta3)) ** (-1)
