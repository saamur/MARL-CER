from typing import Dict

from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

@struct.dataclass
class RCState:
    resistance_nominal: float

    resistance: float
    capacity: float
    i_resistance: float

@struct.dataclass
class ElectricalModelState:
    alpha_fading: float
    beta_fading:float
    r0_nominal: float

    r0: float
    rc: RCState
    ocv_potential: float
    is_active: bool

    v: float
    i: float
    v_rc: float
    q: float

# with fading
class TheveninModel:

    # def __init__(self, components: Dict):
    #     self.r0 = components['r0']
    #     self.rc = freeze(components['rc'])
    #     self.ocv_potential = components['ocv_potential']

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls, components: Dict, v, i, v_rc, is_active, q):
        return ElectricalModelState(components['r0'],
                                    RCState(components['rc']['resistence'],
                                            components['rc']['capacity'],
                                            components['rc']['i_resistance']),
                                    components['ocv_potential'],
                                    is_active, v, i, v_rc, q)       #TODO v_rc?

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step_current_driven(cls, state: ElectricalModelState, i_load:float, dt: float):

        r0 = state.r0
        r1 = state.rc.resistance
        c = state.rc.capacity
        v_ocv = state.ocv_potential

        v_r0 = r0 * i_load
        v_rc = (state.v_rc / dt + i_load /c) / (1/dt + 1/ (c*r1))

        v = v_ocv - v_r0 - v_rc

        i_r1 = v_rc / r1
        # i_c = i_load - i_r1       #TODO non penso serva

        new_q = state.q + jnp.abs(i_load) * dt /3600


        new_state = state.replace(v=v, i=i_load, v_rc=v_rc, q=new_q, rc=state.rc.replace(i_resistence=i_r1))

        return new_state, v, i_load

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_generated_heat(cls, state:ElectricalModelState):
        return (state.r0 * state.i**2 +
                state.rc.resistance * state.rc.i_resistance**2)

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_parameter_fading(cls, state:ElectricalModelState, c_n):
        new_r0 = resistance_fading(r_n=state.r0_nominal, q=state.q, beta=state.beta_fading)
        new_rc_resistance = resistance_fading(r_n=state.rc.resistance_nominal, q=state.q, beta=state.beta_fading)
        new_capacity = capacity_fading(c_n=c_n, q=state.q, alpha=state.alpha_fading)

        new_state = state.replace(r0=new_r0, rc=state.rc.replace(resistance=new_rc_resistance))

        return new_state, new_capacity


def capacity_fading(c_n: float, q: float, alpha: float):
    """
    Capacity fading model depending on the exchange charge in the time interval.

    Parameters
    ----------------
    c_n: Capacity of the battery.
    q: Exchange charge in the time interval.
    alpha: Constant given in configuration file
    """
    return c_n * (1 - alpha * (q ** 0.5))

def resistance_fading(r_n: float, q: float, beta: float):
    """
    Resistor fading model depending on the exchange charge in the time interval.

    Parameters
    ----------------
    r_n: Resistance of the battery.
    q: Exchange charge in the time interval.
    beta: Constant given in configuration file
    """
    return r_n * (1 + beta * (q**0.5))