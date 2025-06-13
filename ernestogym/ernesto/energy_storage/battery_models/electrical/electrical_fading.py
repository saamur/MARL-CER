from typing import Dict

from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

from ernestogym.ernesto.energy_storage.battery_models.electrical.electrical import ElectricalModelState, TheveninModel
from ernestogym.ernesto.energy_storage.battery_models.electrical.ecm_components.resistor import ResistorData

@struct.dataclass
class ElectricalModelFadingState(ElectricalModelState):
    r0_nominal: ResistorData

    alpha_fading: float
    beta_fading:float

    q: float

# with fading
class TheveninFadingModel(TheveninModel):

    @classmethod
    def get_init_state(cls,
                       alpha_fading,
                       beta_fading,
                       components: Dict,
                       inits: Dict,
                       sign_convention: str) -> ElectricalModelFadingState:

        state = super().get_init_state(components, inits, sign_convention)

        return ElectricalModelFadingState(alpha_fading=alpha_fading,
                                          beta_fading=beta_fading,
                                          q=0.,
                                          r0_nominal=state.r0,
                                          r0=state.r0,
                                          rc=state.rc,
                                          ocv_generator=state.ocv_generator,
                                          is_active=state.is_active,
                                          v=state.v,
                                          i=state.i,
                                          v_rc=state.v_rc,
                                          p=state.p)       #TODO v_rc?

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step_current_driven(cls, state: ElectricalModelFadingState, i_load:float, dt: float):

        new_state, v, i_load = super(cls).step_current_driven(state, i_load, dt)

        new_q = state.q + jnp.abs(i_load) * dt / 3600

        new_state = new_state.replace(q=new_q)

        return new_state, v, i_load

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_parameter_fading(cls, state:ElectricalModelFadingState, c_n):
        new_r0_look_up = resistance_fading(r_n=state.r0_nominal.lookup_table, q=state.q, beta=state.beta_fading)
        new_r1_lookup = resistance_fading(r_n=state.rc.r_nominal.lookup_table, q=state.q, beta=state.beta_fading)
        new_capacity = capacity_fading(c_n=c_n, q=state.q, alpha=state.alpha_fading)

        new_state = state.replace(r0=state.r0.replace(lookup_table=new_r0_look_up),
                                  rc=state.rc.replace(r=state.rc.r.replace(lookup_table=new_r1_lookup)))

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
