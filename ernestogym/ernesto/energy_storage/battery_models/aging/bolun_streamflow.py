from typing import Dict

from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

from .streamflow import Streamflow, StreamflowState


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

@struct.dataclass
class DodBolunStressModel:
    k_delta1: float
    k_delta2: float
    k_delta3: float


@struct.dataclass
class BolunStreamflowState:

    init_soh:float
    soh: float
    soc_mean: float
    temp_battery_mean: float
    n_steps: int

    stream_flow_state: StreamflowState

    time_stress_model: TimeStressModel
    soc_stress_model: SocStressModel
    temp_stress_model: TempStressModel
    dod_bolun_stress_model: DodBolunStressModel

    alpha_sei:float
    beta_sei:float



class BolunStreamflowModel:

    @classmethod
    # @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls, components_setting: Dict, stress_models: Dict, temp):

        stream_flow_state = Streamflow.get_init_state()

        time_stress_model = TimeStressModel(k_t=stress_models['time']['k_t'])
        soc_stress_model = SocStressModel(k_soc=stress_models['soc']['k_soc'],
                                          soc_ref=stress_models['soc']['soc_ref'])
        temp_stress_model = TempStressModel(k_temp=stress_models['temperature']['k_temp'])
        dod_bolun_stress_model = DodBolunStressModel(k_delta1=stress_models['dod_bolun']['k_delta1'],
                                                     k_delta2=stress_models['dod_bolun']['k_delta2'],
                                                     k_delta3=stress_models['dod_bolun']['k_delta3'])

        return BolunStreamflowState(init_soh=1.,
                                    soh=1.,
                                    soc_mean=1.,
                                    temp_battery_mean=temp,
                                    n_steps=0,
                                    stream_flow_state=stream_flow_state,
                                    time_stress_model=time_stress_model,
                                    soc_stress_model=soc_stress_model,
                                    temp_stress_model=temp_stress_model,
                                    dod_bolun_stress_model=dod_bolun_stress_model,
                                    alpha_sei=components_setting['SEI']['alpha_sei'],
                                    beta_sei=components_setting['SEI']['beta_sei'])

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_soh(cls, state: BolunStreamflowState, temp_battery, temp_ambient, soc, elapsed_time, is_charging:bool):     #TODO t_ref è temperatura ambiente o un'altra roba?

        new_n_steps = state.n_steps + 1
        new_temp_battery_mean = state.temp_battery_mean + (temp_battery - state.temp_battery_mean) / new_n_steps
        new_soc_mean = state.soc_mean + (soc - state.soc_mean) / new_n_steps

        #calendar aging
        f_cal = (temperature_stress(state.temp_stress_model.k_temp, new_temp_battery_mean, temp_ambient) *
                 soc_stress(state.soc_stress_model.k_soc, new_soc_mean, state.soc_stress_model.soc_ref) *  # fixme siamo sicuri che devo usare la media del soc?
                 time_stress(state.time_stress_model.k_t, elapsed_time))

        expected_end = 0.5 * (is_charging + (1 - 2 * is_charging) * soc)

        ## TODO cose che non so se servono

        new_stream_flow_state = jax.lax.cond((new_n_steps % state.stream_flow_state.reset_every) == 0,
                                             lambda stream_flow_state, soc: Streamflow.get_init_state(soc),
                                             lambda stream_flow_state, soc: stream_flow_state,
                                             state.stream_flow_state, soc)

        new_stream_flow_state = Streamflow.step(new_stream_flow_state, soc, expected_end, temp_battery)

        ranges = new_stream_flow_state.min_max_vals[:, 1] - new_stream_flow_state.min_max_vals[:, 0]

        ranges = jnp.where(ranges == 0, 1e-6, ranges)
        cycle_types = 0.5

        mask = jnp.logical_and(new_stream_flow_state.is_used, new_stream_flow_state.is_valid)

        #call cyclic aging
        f_cyc = cls.compute_cyclic_aging(state,
                                         cycle_type=cycle_types,
                                         cycle_dod=ranges,
                                         avg_cycle_soc=new_stream_flow_state.mean_values,
                                         avg_cycle_temp=new_stream_flow_state.second_signal_means,
                                         mask=mask,
                                         temp_ambient=temp_ambient)

        f_d = f_cal + f_cyc

        deg = jnp.clip(1 - state.alpha_sei * jnp.exp(-state.beta_sei * f_d) - (1 - state.alpha_sei) * jnp.exp(-f_d), 0, 1)

        new_soh = state.init_soh - deg

        new_state = state.replace(n_steps=new_n_steps,
                                  temp_battery_mean=new_temp_battery_mean,
                                  soc_mean=new_soc_mean,
                                  stream_flow_state=new_stream_flow_state,
                                  soh=new_soh)

        return new_state, new_soh

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_cyclic_aging(cls, state: BolunStreamflowState, mask, temp_ambient, cycle_type, cycle_dod, avg_cycle_temp, avg_cycle_soc):
        cyclic_aging = (cycle_type *
                        dod_bolun_stress(cycle_dod, state.dod_bolun_stress_model.k_delta1, state.dod_bolun_stress_model.k_delta2, state.dod_bolun_stress_model.k_delta3) *
                        temperature_stress(state.temp_stress_model.k_temp, avg_cycle_temp, temp_ambient) *
                        soc_stress(state.soc_stress_model.k_soc, avg_cycle_soc, state.soc_stress_model.soc_ref))

        valid_used_cyclic_aging = jnp.where(mask, cyclic_aging, 0)

        return jnp.sum(valid_used_cyclic_aging)


def temperature_stress(k_temp, mean_temp, temp_ref):
    """
    Stress function caused by the temperature, to be computed in Kelvin.
    This k_tempcan be used only for batteries operating above 15°C.

    Inputs:
    :param k_temp: temperature stress coefficient
    :param mean_temp: current battery temperature
    :param temp_ref: ambient temperature
    """
    return jnp.exp(k_temp * (mean_temp - temp_ref) * (temp_ref / mean_temp))
    # return jnp.exp(jnp.float32(k_temp) * (mean_temp - temp_ref) * (temp_ref / mean_temp))

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
    return k_t * t
    # return jnp.float32(k_t) * t


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
    return (k_delta1 * dod ** k_delta2 + k_delta3) ** (-1)
    # return (jnp.float32(k_delta1) * dod ** jnp.float32(k_delta2) + jnp.float32(k_delta3)) ** (-1)
