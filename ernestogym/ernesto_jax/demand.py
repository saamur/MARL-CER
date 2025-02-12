import numpy as np
from flax import struct
from functools import partial
import jax
import jax.numpy as jnp
from .utils import change_timestep_array

@struct.dataclass
class DemandData:
    data: jnp.ndarray
    timestep: int

    min: float
    max: float

class Demand:

    @classmethod
    def build_demand_data(cls, demand: jnp.ndarray, in_timestep: int, out_timestep: int, max_length: int) -> DemandData:

        assert len(demand) * in_timestep >= max_length

        data = change_timestep_array(demand[:np.ceil(max_length / in_timestep)], in_timestep, out_timestep, 'sum')

        data = jnp.array(data[:max_length // out_timestep])

        return DemandData(data=data,
                          timestep=out_timestep,
                          max=jnp.max(data),
                          min=jnp.min(data))

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_demand(cls, demand_data: DemandData, t: int) -> jnp.ndarray:
        return demand_data.data[t // demand_data.timestep]

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def is_run_out_of_data(cls, demand_data: DemandData, t: int) -> bool:
        return t // demand_data.timestep >= len(demand_data.data)