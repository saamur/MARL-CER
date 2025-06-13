import os

import numpy as np
import pandas as pd
import jax.numpy as jnp


def read_csv(csv_file: str) -> pd.DataFrame:
    """
    Read data from csv files
    """
    # Check file existence
    if not os.path.isfile(csv_file):
        raise FileNotFoundError("The specified file '{}' doesn't not exist.".format(csv_file))

    df = None
    try:
        df = pd.read_csv(csv_file)
    except Exception as err:
        print("Error during the loading of '{}':".format(csv_file), type(err).__name__, "-", err)

    return df

def change_timestep_array(array: jnp.ndarray, in_timestep: int, out_timestep: int, agg_func: str) -> jnp.ndarray:

    lcm = np.lcm(in_timestep, out_timestep)

    repeated = jnp.repeat(array, lcm // out_timestep)
    repeated = repeated[:len(repeated) // (lcm//in_timestep) * (lcm//in_timestep)]

    array = jnp.reshape(repeated, shape=(-1, lcm // in_timestep))

    if agg_func == 'sum':
        return jnp.sum(array, axis=1) / (lcm // out_timestep)
    elif agg_func == 'mean':
        return jnp.mean(array, axis=1)
    else:
        raise ValueError("Invalid aggregation function '{}'".format(agg_func))
