from typing import Dict

from flax import struct
from functools import partial
import jax
import jax.numpy as jnp

from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.ecm_components.resistor import Resistor, ResistorData
from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.ecm_components.rc_parallel import RCParallel, RCData
from ernestogym.ernesto_jax.energy_storage.battery_models.electrical.ecm_components.ocv_generator import OCVGenerator, OCVData

@struct.dataclass
class RCState:
    resistance_nominal: float

    resistance: float
    capacity: float
    i_resistance: float

@struct.dataclass
class ElectricalModelState:
    r0_nominal: ResistorData
    r0: ResistorData

    rc: RCData
    ocv_generator: OCVData
    is_active: bool

    v: float
    i: float
    p: float
    v_rc: float

# with fading
class TheveninModel:

    @classmethod
    def get_init_state(cls,
                       components: Dict,
                       inits: Dict,
                       sign_convention: str):

        r0_nominal = Resistor.get_initial_state(components['r0'])

        rc = RCParallel.get_initial_state(components['r1'], components['c'])

        ocv_generator = OCVGenerator.get_initial_state(components['v_ocv'])

        return ElectricalModelState(r0_nominal=r0_nominal,
                                    r0=r0_nominal,
                                    rc=rc,
                                    ocv_generator=ocv_generator,
                                    is_active= sign_convention == 'active',
                                    v=inits['voltage'],
                                    i=inits['current'],
                                    v_rc=0.,
                                    p=0.)       #TODO v_rc?

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step_current_driven(cls, state: ElectricalModelState, i_load:float, temp: float, soc: float, dt: float):

        r0 = Resistor.get_resistence(state.r0, temp, soc)
        r1 = RCParallel.get_resistence(state.rc, temp, soc)
        c = RCParallel.get_capacity(state.rc, temp, soc)
        v_ocv = OCVGenerator.get_potential(state.ocv_generator, temp, soc)

        i_load = jnp.where(state.is_active, i_load, -i_load)

        v_r0 = r0 * i_load

        #######################

        # v_rc = (state.v_rc / dt + i_load /c) / (1/dt + 1/ (c*r1))

        # v = v_ocv - v_r0 - v_rc
        #
        # i_r1 = v_rc / r1

        ###############################

        e = jnp.exp(-dt/(r1*c))

        v_rc = r1 * i_load * (1 - e) + e * state.v_rc          #1/exp * (r1 * i_load * (exp - 1) + state.v_rc)

        v = v_ocv - v_r0 - v_rc

        i_r1 = v_rc / r1

        # jax.debug.print('voltage: {v}', v=v, ordered=True)

        power = v * i_load

        power = jnp.where(state.is_active, power, -power)

        new_state = state.replace(v=v, i=i_load, v_rc=v_rc, rc=state.rc.replace(i_resistance=i_r1), p=power)

        return new_state, v, i_load

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def step_power_driven(cls, state: ElectricalModelState, p_load: float, temp: float, soc: float, dt: float):

        return cls.step_current_driven(state, p_load/state.v, temp, soc, dt)

    @classmethod
    @partial(jax.jit, static_argnums=[0])
    def compute_generated_heat(cls, state:ElectricalModelState, temp: float, soc: float):
        return (Resistor.get_resistence(state.r0, temp=temp, soc=soc) * state.i**2 +
                Resistor.get_resistence(state.rc.r, temp=temp, soc=soc) * state.rc.i_resistance**2)