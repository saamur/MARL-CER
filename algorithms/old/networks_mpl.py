import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from flax.nnx.nn.initializers import constant, orthogonal
import numpy as np
from typing import Sequence
import distrax

from algorithms.normalization_custom import RunningNorm


class RECMLP(nnx.Module):
    def __init__(self,
                 obs_keys:tuple,
                 obs_is_local:dict,
                 num_battery_agents: int,
                 activation: str,
                 rngs,
                 net_arch: tuple=(),
                 non_shared_net_arch_before: tuple=(),
                 non_shared_net_arch_after: tuple=(),
                 passive_houses: bool=False,
                 normalize:bool=True,
                 is_obs_normalizable: Sequence[bool] = None):

        self.obs_keys = obs_keys

        print('RECMLP', self.obs_keys)

        self.obs_is_local = obs_is_local
        self.num_battery_agents = num_battery_agents

        in_features = len(self.obs_keys)

        self.normalize = normalize

        self.normalize = normalize
        if self.normalize:
            print('norm rec')
            self.norm_layer = RunningNorm(num_features=in_features, use_bias=False, use_scale=False, rngs=rngs)

        self.passive_houses = passive_houses

        net_arch = tuple(net_arch)

        activation = activation_from_name(activation)

        @nnx.split_rngs(splits=num_battery_agents)
        @nnx.vmap(in_axes=(None, None, None, None, 0))
        def stacked_layer(in_features, out_features, kernel_init, bias_init, rngs):
            return nnx.Linear(in_features, out_features, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)

        def build_layers(net_arch_before, net_arch, net_arch_after, in_features):

            layers_before = []

            last_len = in_features

            if len(net_arch_before) != 0:
                net_arch_before = (in_features,) + net_arch_before
                last_len = net_arch_before[-1]

                for i in range(len(net_arch_before) - 1):
                    layers_before.append(stacked_layer(net_arch_before[i], net_arch_before[i+1],
                                                                           kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
                    layers_before.append(activation)

                if len(net_arch) == 0 and len(net_arch_after) == 0:
                    layers_before.append(stacked_layer(net_arch_before[-1], 1,
                                                                           kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))

            layers = []
            if len(net_arch) != 0:
                net_arch = (last_len,) + net_arch
                last_len = net_arch[-1]
                for i in range(len(net_arch) - 1):
                    layers.append(
                        nnx.Linear(net_arch[i], net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)),
                                   bias_init=constant(0.), rngs=rngs))
                    layers.append(activation)
                if len(net_arch_after) == 0:
                    layers.append(
                        nnx.Linear(net_arch[-1], 1, kernel_init=orthogonal(0.01), bias_init=constant(0.),
                                   rngs=rngs))

            layers_after = []
            if len(net_arch_after) != 0:
                net_arch_after = (last_len,) + net_arch_after
                for i in range(len(net_arch_after) - 1):
                    layers_after.append(stacked_layer(net_arch_after[i], net_arch_after[i+1],
                                                                          kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
                    layers_after.append(activation)
                layers_after.append(stacked_layer(net_arch_after[-1], 1,
                                                                      kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))

            return layers_before, layers, layers_after

        self.layers_before, self.layers, self.layers_after = build_layers(non_shared_net_arch_before, net_arch, non_shared_net_arch_after, in_features)

        self.activation = activation

    def __call__(self, obs):
        data = self.prepare_data(obs)

        print('dataaa mlp', data.shape)

        logit = data

        logit = self.call_non_shared_layers(self.layers_before, logit)

        for layer in self.layers:
            logit = layer(logit)

        logit = self.call_non_shared_layers(self.layers_after, logit)
        logit = logit.squeeze(axis=-1)
        print('logit', logit.shape)

        vec = nnx.softmax(logit)

        return vec

    def call_non_shared_layers(self, layers, data):
        @nnx.split_rngs(splits=self.num_battery_agents)
        @nnx.vmap(in_axes=(0, -2), out_axes=-2)
        def compute_layer(lay, data):
            return lay(data)

        out = data
        for layer in layers:
            if isinstance(layer, nnx.Module):
                out = compute_layer(layer, out)
            else:
                out = layer(out)

        return out

    def prepare_data(self, obs):

        local_obs = [obs[key] for key in self.obs_keys if self.obs_is_local[key]]
        global_obs = [obs[key] for key in self.obs_keys if not self.obs_is_local[key]]

        local_data = jnp.stack(local_obs, axis=-1)
        global_data = jnp.stack(global_obs, axis=-1)
        global_data = jnp.expand_dims(global_data, -2)
        global_data = jnp.repeat(global_data, self.num_battery_agents, axis=-2)

        data_act = jnp.concatenate((global_data, local_data), axis=-1)

        if self.normalize:
            data_act = self.norm_layer(data_act)

        return data_act

def activation_from_name(name: str):
    name = name.lower()
    if name == 'relu':
        return nnx.relu
    elif name == 'tanh':
        return nnx.tanh
    elif name == 'sigmoid':
        return nnx.sigmoid
    elif name == 'leaky_relu':
        return nnx.leaky_relu
    elif name == 'swish':
        return nnx.swish
    elif name == 'elu':
        return nnx.elu
    else:
        raise ValueError("'activation' must be 'relu' or 'tanh'")
