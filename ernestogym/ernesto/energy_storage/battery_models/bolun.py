from typing import Dict

from flax import struct
from flax.core.frozen_dict import freeze
from functools import partial
import jax
import jax.numpy as jnp

# @struct.dataclass
# class RCState:
#     resistance: float
#     capacity: float


@struct.dataclass
class StreamflowState:
    is_init: bool
    cycle_k: int
    last_soc_value: float

    is_direction_up: jnp.ndarray
    is_used: jnp.ndarray
    is_valid: jnp.ndarray
    min_max_vals: jnp.ndarray
    mean_values: jnp.ndarray
    second_signal_means: jnp.ndarray
    number_of_samples: jnp.ndarray
    start_cycles: jnp.ndarray

    iteration: int
    reset_every: int


class Streamflow:

    @classmethod
    @partial(jax.jit, static_argnums=[0, 2])
    def get_init_state(cls, init_soc=1., expected_cycle_num=3000):
        init_state = StreamflowState(is_init=True,
                                     cycle_k=-1,
                                     last_soc_value=init_soc,
                                     is_direction_up=jnp.zeros(expected_cycle_num, dtype=bool),
                                     is_used=jnp.zeros(expected_cycle_num, dtype=bool),
                                     is_valid=jnp.zeros(expected_cycle_num, dtype=bool),
                                     min_max_vals=jnp.zeros((expected_cycle_num, 2)),
                                     mean_values=jnp.zeros(expected_cycle_num),
                                     second_signal_means=jnp.zeros(expected_cycle_num),
                                     number_of_samples=jnp.zeros(expected_cycle_num, dtype=int),
                                     start_cycles=jnp.zeros(expected_cycle_num, dtype=int),
                                     iteration=0,
                                     reset_every=50000)
        return init_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step(cls, state: StreamflowState, actual_soc_value, expected_end, second_signal_value) -> StreamflowState:
        change_direction = jnp.logical_or(state.is_init,
                                          jnp.logical_or(jnp.logical_and(state.is_direction_up[state.cycle_k],
                                                                         actual_soc_value < state.last_soc_value),
                                                         jnp.logical_and(
                                                             jnp.logical_not(state.is_direction_up[state.cycle_k]),
                                                             actual_soc_value > state.last_soc_value)))
        new_is_init = False

        new_state = state.replace(is_init=new_is_init)

        # TODO parte grossa
        new_state = jax.lax.cond(change_direction,
                                 cls.create_new_cycle,
                                 cls.update_existent_cycle,
                                 new_state, actual_soc_value, second_signal_value, expected_end)

        new_iteration = state.iteration + 1

        new_state = new_state.replace(is_init=new_is_init, last_soc_value=actual_soc_value, iteration=new_iteration)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def create_new_cycle(cls, state: StreamflowState, value: float, second_signal_value: float, expected_end: float):
        new_cycle_k = jnp.sum(state.is_used)

        # FIXME COME FACCIAMO SE VA OLTRE IL LIMITE?

        new_is_valid = state.is_valid.at[new_cycle_k].set(True)
        new_is_used = state.is_used.at[new_cycle_k].set(True)
        new_is_direction_up = state.is_direction_up.at[new_cycle_k].set(value > state.last_soc_value)

        min_val, max_val = jax.lax.min(value, state.last_soc_value), jax.lax.max(value, state.last_soc_value)
        new_min_max_vals = state.min_max_vals.at[new_cycle_k].set((min_val, max_val))

        new_mean_values = state.mean_values.at[new_cycle_k].set(value)
        new_second_signal_means = state.second_signal_means.at[new_cycle_k].set(second_signal_value)

        new_number_of_samples = state.number_of_samples.at[new_cycle_k].set(1)
        new_start_cycles = state.start_cycles.at[new_cycle_k].set(state.iteration)

        new_state = state.replace(is_valid=new_is_valid,
                                  is_used=new_is_used,
                                  is_direction_up=new_is_direction_up,
                                  min_max_vals=new_min_max_vals,
                                  mean_values=new_mean_values,
                                  second_signal_means=new_second_signal_means,
                                  number_of_samples=new_number_of_samples,
                                  start_cycles=new_start_cycles)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def update_existent_cycle(cls, state: StreamflowState, value: float, second_signal_value: float,
                              expected_end: float):
        valid_used_correct_directions = jnp.logical_and(jnp.logical_and(state.is_used, state.is_valid),
                                                        state.is_direction_up == state.is_direction_up[state.cycle_k])

        indices = cls._get_indices_by_direction(direction=state.is_direction_up[state.cycle_k],
                                                min_max_vals=state.min_max_vals,
                                                last_value=state.last_soc_value,
                                                value=value,
                                                indices_range=valid_used_correct_directions)

        expected_end_indices = cls._get_indices_by_direction(direction=state.is_direction_up[state.cycle_k],
                                                             min_max_vals=state.min_max_vals,
                                                             last_value=state.last_soc_value,
                                                             value=expected_end,
                                                             indices_range=valid_used_correct_directions)

        aux_ind = jnp.logical_and(valid_used_correct_directions, indices)

        def when_any_exp_end_ind(state: StreamflowState, aux_ind):
            to_invalid = aux_ind
            return jnp.where(to_invalid, False, state.is_valid), state.cycle_k

        def when_no_exp_end_ind(state: StreamflowState, aux_ind: jnp.ndarray):
            new_is_valid = state.is_valid.at[state.cycle_k].set(False)
            new_cycle_k = jax.lax.cond(state.is_direction_up[state.cycle_k],
                                       lambda mask, min_max_vals: jnp.nanargmax(
                                           jnp.where(mask, min_max_vals[:, 1], jnp.nan)),
                                       lambda mask, min_max_vals: jnp.nanargmin(
                                           jnp.where(mask, min_max_vals[:, 0], jnp.nan)),
                                       aux_ind, state.min_max_vals)
            # indices = jnp.arange(len(new_is_valid))
            # fixme nel codice di Davide è più complesso ma in teoria dovrebbe essere la stessa cosa
            to_invalid = aux_ind
            new_is_valid = jnp.where(to_invalid, False, new_is_valid)
            new_is_valid = new_is_valid.at[new_cycle_k].set(True)
            return new_is_valid, new_cycle_k

        new_is_valid, new_cycle_k = jax.lax.cond(indices.any(),
                                                 lambda state, aux_ind: jax.lax.cond(expected_end_indices.any(),
                                                                                     when_any_exp_end_ind,
                                                                                     when_no_exp_end_ind, state,
                                                                                     aux_ind),
                                                 lambda state, aux_ind: (state.is_valid, state.cycle_k),
                                                 state, aux_ind)
        # new_is_valid, new_cycle_k = jax.lax.cond(expected_end_indices.any(), when_any_exp_end_ind, when_no_exp_end_ind, state, aux_ind)

        new_mean_values = state.mean_values.at[new_cycle_k].set(state.mean_values[new_cycle_k] +
                                                                (value - state.mean_values[new_cycle_k]) /
                                                                (state.number_of_samples[new_cycle_k] + 1))

        new_second_signal_values = state.second_signal_means.at[new_cycle_k].set(
            state.second_signal_means[new_cycle_k] +
            (second_signal_value - state.second_signal_means[new_cycle_k]) /
            (state.number_of_samples[new_cycle_k] + 1))

        new_min_max_vals = state.min_max_vals.at[new_cycle_k].set(
            (jax.lax.min(value, state.min_max_vals[new_cycle_k, 0]),
             jax.lax.max(value, state.min_max_vals[new_cycle_k, 1])))

        new_number_of_samples = state.number_of_samples.at[new_cycle_k].set(state.number_of_samples[new_cycle_k] + 1)
        new_state = state.replace(cycle_k=new_cycle_k,
                                  is_valid=new_is_valid,
                                  min_max_vals=new_min_max_vals,
                                  mean_values=new_mean_values,
                                  second_signal_means=new_second_signal_values,
                                  number_of_samples=new_number_of_samples)

        return new_state

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def _get_indices_by_direction(cls, direction: bool, min_max_vals, last_value, value: float,
                                  indices_range: jnp.array):
        indices = jax.lax.cond(direction,
                               lambda min_max_vals, last_value, value: jnp.logical_and(
                                   min_max_vals[:, 1] > last_value,
                                   min_max_vals[:, 1] < value),
                               lambda min_max_vals, last_value, value: jnp.logical_and(
                                   min_max_vals[:, 0] < last_value,
                                   min_max_vals[:, 0] > value),
                               min_max_vals, last_value, value)

        return jnp.logical_and(indices, indices_range)


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
class AgingModelState:

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



class BolunAgingModel:

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

        return AgingModelState(init_soh=1.,
                               soh=1.,
                               soc_mean=1,
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
    def compute_soh(cls, state: AgingModelState, temp_battery, temp_ambient, soc, elapsed_time, is_charging:bool):     #TODO t_ref è temperatura ambiente o un'altra roba?

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
    def compute_cyclic_aging(cls, state: AgingModelState, mask, temp_ambient, cycle_type, cycle_dod, avg_cycle_temp, avg_cycle_soc):
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
