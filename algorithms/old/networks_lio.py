import numpy as np
import jax.numpy as jnp
from flax import nnx

from flax.nnx.nn.initializers import constant, orthogonal, xavier_normal

from algorithms.normalization_custom import RunningNorm

class IncentiveNetwork(nnx.Module):

    def __init__(self,
                 obs_keys: tuple,
                 num_agents: int,
                 activation: str,
                 net_arch: tuple,
                 rngs,
                 normalize: bool = True):

        self.obs_keys = tuple(obs_keys)

        in_feat = len(obs_keys) + num_agents - 1

        self.normalize = normalize
        if normalize:
            self.norm_layer = RunningNorm(num_features=in_feat, use_bias=False, use_scale=False, rngs=rngs)

        activation = activation_from_name(activation)

        net_arch = (in_feat,) + net_arch + (num_agents-1,)

        self.layers = []

        for i in range(len(net_arch)-2):
            self.layers.append(nnx.Linear(net_arch[i], net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.layers.append(activation)
        self.layers.append(nnx.Linear(net_arch[-2], net_arch[-1], kernel_init=orthogonal(0.1), bias_init=constant(0.), rngs=rngs))

        # for i in range(len(net_arch)-1):
        #     self.layers.append(nnx.Linear(net_arch[i], net_arch[i+1], kernel_init=constant(0.), bias_init=constant(0.), rngs=rngs))
        #     self.layers.append(activation)
        # self.layers.pop()

        # for i in range(len(net_arch)-1):
        #     self.layers.append(nnx.Linear(net_arch[i], net_arch[i+1], kernel_init=xavier_normal(), bias_init=constant(0.), rngs=rngs))
        #     self.layers.append(activation)
        # self.layers.pop()

    def __call__(self, obs, others_actions):
        data = self._prepare_data(obs, others_actions)

        for layer in self.layers:
            data = layer(data)

        return data

    def _prepare_data(self, obs, others_actions):
        data = jnp.stack([obs[key] for key in self.obs_keys], axis=-1)
        data = jnp.concat((data, others_actions), axis=-1)

        return data

class IncentiveNetworkPercentage(nnx.Module):

    def __init__(self,
                 obs_keys: tuple,
                 num_agents: int,
                 activation: str,
                 net_arch: tuple,
                 rngs,
                 normalize: bool = True):

        self.obs_keys = tuple(obs_keys)

        in_feat = len(obs_keys) + num_agents

        self.normalize = normalize
        if normalize:
            self.norm_layer = RunningNorm(num_features=in_feat, use_bias=False, use_scale=False, rngs=rngs)

        activation = activation_from_name(activation)

        net_arch = (in_feat,) + net_arch + (num_agents,)

        self.layers = []

        for i in range(len(net_arch)-2):
            self.layers.append(nnx.Linear(net_arch[i], net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.layers.append(activation)
        self.layers.append(nnx.Linear(net_arch[-2], net_arch[-1], kernel_init=orthogonal(0.1), bias_init=constant(0.), rngs=rngs))

        # for i in range(len(net_arch)-1):
        #     self.layers.append(nnx.Linear(net_arch[i], net_arch[i+1], kernel_init=constant(0.), bias_init=constant(0.), rngs=rngs))
        #     self.layers.append(activation)
        # self.layers.pop()

        # for i in range(len(net_arch)-1):
        #     self.layers.append(nnx.Linear(net_arch[i], net_arch[i+1], kernel_init=xavier_normal(), bias_init=constant(0.), rngs=rngs))
        #     self.layers.append(activation)
        # self.layers.pop()

    def __call__(self, obs, others_actions):
        data = self._prepare_data(obs, others_actions)

        for layer in self.layers:
            data = layer(data)

        data = nnx.softmax(data, axis=-1)

        return data

    def _prepare_data(self, obs, others_actions):
        data = jnp.stack([obs[key] for key in self.obs_keys], axis=-1)
        data = jnp.concat((data, others_actions), axis=-1)

        return data

class StackedIncentiveNetwork(IncentiveNetwork):

    def __init__(self,
                 num_networks: int,
                 obs_keys: tuple,
                 num_agents: int,
                 activation: str,
                 net_arch: tuple,
                 rngs,
                 normalize: bool = True):

        self.num_networks = num_networks

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, None, None, None, None, 0, None))
        def vmapped_fn(self, obs_keys: tuple, num_agents: int, activation: str, net_arch: tuple, rngs, normalize: bool):
            super(StackedIncentiveNetwork, self).__init__(obs_keys, num_agents, activation, net_arch, rngs, normalize)

        vmapped_fn(self, obs_keys, num_agents, activation, net_arch, rngs, normalize)

    def __call__(self, obs, others_actions):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, obs, others_actions):
            return super(StackedIncentiveNetwork, self).__call__(obs, others_actions)

        return vmapped_fn(self, obs, others_actions)

class StackedIncentiveNetworkPercentage(IncentiveNetworkPercentage):

    def __init__(self,
                 num_networks: int,
                 obs_keys: tuple,
                 num_agents: int,
                 activation: str,
                 net_arch: tuple,
                 rngs,
                 normalize: bool = True):

        self.num_networks = num_networks

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, None, None, None, None, 0, None))
        def vmapped_fn(self, obs_keys: tuple, num_agents: int, activation: str, net_arch: tuple, rngs, normalize: bool):
            super(StackedIncentiveNetworkPercentage, self).__init__(obs_keys, num_agents, activation, net_arch, rngs, normalize)

        vmapped_fn(self, obs_keys, num_agents, activation, net_arch, rngs, normalize)

    def __call__(self, obs, others_actions):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, obs, others_actions):
            return super(StackedIncentiveNetworkPercentage, self).__call__(obs, others_actions)

        return vmapped_fn(self, obs, others_actions)


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
