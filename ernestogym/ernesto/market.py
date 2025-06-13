import numpy as np
from flax import struct
from functools import partial
import jax
import jax.numpy as jnp
from .utils import change_timestep_array

SECONDS_PER_DAY = 60 * 60 * 24

@struct.dataclass
class BuyingPriceData:
    data: jnp.ndarray
    timestep: int
    circular: bool

    min: float
    max: float

class BuyingPrice:

    @classmethod
    def build_buying_price_data(cls, buying_price: jnp.ndarray, in_timestep: int, out_timestep: int, max_length: int, circular: bool = False) -> BuyingPriceData:

        if circular:
            buying_price = np.tile(buying_price, np.ceil(max_length / (len(buying_price) * in_timestep)).astype(int))
        else:
            assert len(buying_price) * in_timestep >= max_length

        data = change_timestep_array(buying_price[:np.ceil(max_length / in_timestep).astype(int)], in_timestep, out_timestep, 'mean')

        data = jnp.array(data[:max_length // out_timestep])

        return BuyingPriceData(data=data,
                               timestep=out_timestep,
                               circular=circular,
                               max = jnp.max(data),
                               min = jnp.min(data))


    @classmethod
    def build_circular_buying_price_data_from_time_bands(cls, bands_boundaries: list, bands_prices: list, out_timestep: int, max_length: int) -> BuyingPriceData:
        assert len(bands_boundaries) + 1 == len(bands_prices)

        data = jnp.empty(shape=(SECONDS_PER_DAY,))

        bands_boundaries = [0] + bands_boundaries + [SECONDS_PER_DAY]

        for i in range(len(bands_prices)):
            data = data.at[bands_boundaries[i]:bands_boundaries[i+1]].set(bands_prices[i])

        return cls.build_buying_price_data(data, 1, out_timestep, max_length, circular=True)


    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_buying_price(cls, price_data: BuyingPriceData, t: int) -> jnp.ndarray:
        index =  jax.lax.cond(price_data.circular,
                              lambda: (t / price_data.timestep) % len(price_data.data),
                              lambda: t / price_data.timestep)

        index = jnp.astype(index, int)

        return price_data.data[index]

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def is_run_out_of_data(cls, buying_price_data: BuyingPriceData, t: int):
        return jnp.logical_and(jnp.logical_not(buying_price_data.circular),
                               (t // buying_price_data.timestep >= len(buying_price_data.data)))


@struct.dataclass
class SellingPriceData:
    data: jnp.ndarray
    timestep: int

    min: float
    max: float

class SellingPrice:

    @classmethod
    def build_selling_price_data(cls, selling_price: jnp.ndarray, in_timestep: int, out_timestep: int, max_length: int) -> SellingPriceData:

        assert len(selling_price) * in_timestep >= max_length

        data = change_timestep_array(selling_price[:np.ceil(max_length / in_timestep).astype(int)], in_timestep, out_timestep, 'mean')

        data = jnp.array(data[:max_length // out_timestep])

        return SellingPriceData(data=data,
                                timestep=out_timestep,
                                max=jnp.max(data),
                                min=jnp.min(data))

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_selling_price(cls, demand_data: SellingPriceData, t: int) -> jnp.ndarray:
        return demand_data.data[jnp.astype(t / demand_data.timestep, int)]

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def is_run_out_of_data(cls, selling_price_data: SellingPriceData, t: int) -> bool:
        return t // selling_price_data.timestep >= len(selling_price_data.data)
