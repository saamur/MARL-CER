import numpy as np
from flax import struct
from functools import partial
import jax
import jax.numpy as jnp
from .utils import change_timestep_array

@struct.dataclass
class TemperatureData:
    data: jnp.ndarray
    timestep: int

    min: float
    max: float

class AmbientTemperature:

    @classmethod
    def build_generation_data(cls, temperature: jnp.ndarray, in_timestep: int, out_timestep: int, max_length: int) -> TemperatureData:

        assert len(temperature) * in_timestep >= max_length

        data = change_timestep_array(temperature[:np.ceil(max_length / in_timestep).astype(int)], in_timestep, out_timestep, 'mean')

        data = jnp.array(data[:max_length // out_timestep])

        return TemperatureData(data=data,
                               timestep=out_timestep,
                               max=jnp.max(data),
                               min=jnp.min(data))

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_amb_temperature(cls, temperature_data: TemperatureData, t: int) -> jnp.ndarray:
        return temperature_data.data[jnp.astype(t / temperature_data.timestep, int)]

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def is_run_out_of_data(cls, temperature_data: TemperatureData, t: int) -> bool:
        return t // temperature_data.timestep >= len(temperature_data.data)