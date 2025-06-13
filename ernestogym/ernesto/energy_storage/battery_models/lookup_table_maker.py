import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

params_csv_folder = 'ernestogym/ernesto/data/battery/params_kelvin/'

def build_grid_for_lookup(data):
    if isinstance(data, str):
        data = pd.read_csv(os.path.join(params_csv_folder, data))
        data = data.to_numpy()

    assert data.shape[1] == 2 or data.shape[1] == 3

    if data.shape[1] == 2:
        xs = np.unique(data[:, 0])
        x_step = xs[1] - xs[0]
        x_ref = xs[0]

        assert data.shape[0] == len(xs)

        assert (np.abs(xs[1:] - xs[:-1] - x_step) < 1e-6).all()
        array = np.empty((len(xs),))
        array[np.rint((data[:, 0]-x_ref) / x_step).astype(int)] = data[:, 1]

        return array, x_ref, x_step

    elif data.shape[1] == 3:
        xs = np.unique(data[:, 0])
        ys = np.unique(data[:, 1])

        assert len(np.unique(data[:, 0:2], axis=0)) == len(xs) * len(ys)

        x_step = xs[1] - xs[0]
        y_step = ys[1] - ys[0]
        x_ref = xs[0]
        y_ref = ys[0]

        assert (np.abs(xs[1:] - xs[:-1] - x_step) < 1e-6).all()
        assert (np.abs(ys[1:] - ys[:-1] - y_step) < 1e-6).all()

        matrix = np.empty((len(xs), len(ys)))

        matrix[np.rint((data[:, 0]-x_ref) / x_step).astype(int), np.rint((data[:, 1]-y_ref) / y_step).astype(int)] = data[:, 2]

        return matrix, x_ref, y_ref, x_step, y_step

    else:
        raise NotImplementedError('Only single or two input functions interpolation is supported')

def get_interpolation_2d(lookup_table, x_ref, y_ref, x_step, y_step, x, y):
    norm = jnp.array([x_step, y_step])
    point = jnp.array([x-x_ref, y-y_ref])
    coord = point / norm
    coord = coord[:, None]

    return jax.scipy.ndimage.map_coordinates(lookup_table, coord, order=1, mode='nearest')[0]

def get_interpolation_1d(lookup_table, x_ref, x_step, x):
    point = jnp.array([x-x_ref])
    coord = point / x_step
    coord = coord[:, None]

    return jax.scipy.ndimage.map_coordinates(lookup_table, coord, order=1, mode='nearest')[0]