import numpy as np
from flax import struct
from functools import partial
import jax
import jax.numpy as jnp
from .utils import change_timestep_array

@struct.dataclass
class GenerationData:
    data: jnp.ndarray
    timestep: int

    min: float
    max: float

class Generation:

    @classmethod
    def build_generation_data(cls, generation: jnp.ndarray, in_timestep: int, out_timestep: int, max_length: int) -> GenerationData:

        assert len(generation) * in_timestep >= max_length

        data = change_timestep_array(generation[:np.ceil(max_length / in_timestep).astype(int)], in_timestep, out_timestep, 'sum')

        data = jnp.array(data[:max_length // out_timestep])

        return GenerationData(data=data,
                              timestep=out_timestep,
                              max=jnp.max(data),
                              min=jnp.min(data))

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_generation(cls, generation_data: GenerationData, t: int) -> jnp.ndarray:
        return generation_data.data[jnp.astype(t / generation_data.timestep, int)]

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def is_run_out_of_data(cls, generation_data: GenerationData, t: int) -> bool:
        return t // generation_data.timestep >= len(generation_data.data)