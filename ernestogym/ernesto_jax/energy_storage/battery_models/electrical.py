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
    r0_nominal: float
    r0: float

    rc: RCState
    ocv_potential: float
    is_active: bool

    v: float
    i: float
    p: float
    v_rc: float

# with fading
class TheveninModel:

    @classmethod
    # @partial(jax.jit, static_argnums=[0])
    def get_init_state(cls,
                       components: Dict,
                       inits: Dict,
                       sign_convention: str):

        assert components['r0']['selected_type'] == 'scalar'
        r0 = components['r0']['scalar']
        r0_nominal = r0

        assert components['r1']['selected_type'] == 'scalar'
        assert components['c']['selected_type'] == 'scalar'
        rc = RCState(resistance_nominal=components['r1']['scalar'],
                     resistance=components['r1']['scalar'],
                     capacity=components['c']['scalar'],
                     i_resistance=0.)    # FIXME 0 giusto?

        assert components['v_ocv']['selected_type'] == 'scalar'
        ocv_potential = components['v_ocv']['scalar']

        return ElectricalModelState(r0_nominal=r0_nominal,
                                    r0=r0,
                                    rc=rc,
                                    ocv_potential=ocv_potential,
                                    is_active= sign_convention == 'active',
                                    v=inits['voltage'],
                                    i=inits['current'],
                                    v_rc=0.,
                                    p=0.)       #TODO v_rc?

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step_current_driven(cls, state: ElectricalModelState, i_load:float, dt: float):

        r0 = state.r0
        r1 = state.rc.resistance
        c = state.rc.capacity
        v_ocv = state.ocv_potential

        i_load = jnp.where(state.is_active, i_load, -i_load)

        v_r0 = r0 * i_load
        v_rc = (state.v_rc / dt + i_load /c) / (1/dt + 1/ (c*r1))

        v = v_ocv - v_r0 - v_rc

        i_r1 = v_rc / r1
        # i_c = i_load - i_r1       #TODO non penso serva

        power = v * i_load

        power = jnp.where(state.is_active, power, -power)

        new_state = state.replace(v=v, i=i_load, v_rc=v_rc, rc=state.rc.replace(i_resistance=i_r1), p=power)

        return new_state, v, i_load

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_generated_heat(cls, state:ElectricalModelState):
        return (state.r0 * state.i**2 +
                state.rc.resistance * state.rc.i_resistance**2)