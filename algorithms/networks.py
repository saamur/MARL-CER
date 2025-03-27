import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from flax.nnx.nn.initializers import constant, orthogonal
import numpy as np
from typing import Sequence
import distrax

from algorithms.normalization_custom import RunningNorm
import algorithms.utils as utils


class ActorCritic(nnx.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, rngs, net_arch: list=None, act_net_arch: list=None, cri_net_arch: list=None, add_logistic_to_actor: bool = False, normalize: bool = False, is_feature_normalizable: Sequence[bool] = None):

        if act_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            act_net_arch = net_arch
        if cri_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            cri_net_arch = net_arch

        act_net_arch = list(act_net_arch)
        cri_net_arch = list(cri_net_arch)

        self.normalize = normalize
        if normalize:
            print('norm batt')
            # self.norm_layer = nnx.BatchNorm(num_features=in_features, use_bias=False, use_scale=False, rngs=rngs)
            self.norm_layer = RunningNorm(num_features=in_features, use_bias=False, use_scale=False, rngs=rngs)

        activation = utils.activation_from_name(activation)

        act_net_arch = [in_features] + act_net_arch + [out_features]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.1), bias_init=constant(0.), rngs=rngs))
        if add_logistic_to_actor:
            self.act_layers.append(nnx.sigmoid)

        self.log_std = nnx.Param(jnp.zeros(out_features))# - 1.)

        cri_net_arch = [in_features] + cri_net_arch + [1]

        self.cri_layers = []
        for i in range(len(cri_net_arch) - 2):
            self.cri_layers.append(nnx.Linear(cri_net_arch[i], cri_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.cri_layers.append(activation)
        self.cri_layers.append(nnx.Linear(cri_net_arch[-2], cri_net_arch[-1], kernel_init=orthogonal(1.), bias_init=constant(0.), rngs=rngs))

        # self.act_dense1 = nnx.Linear(in_features, 64, kernel_init=glorot_normal(), bias_init=constant(0.), rngs=rngs)
        # self.act_dense2 = nnx.Linear(64, 64, kernel_init=glorot_normal(), bias_init=constant(0.), rngs=rngs)
        # self.act_dense3 = nnx.Linear(64, out_features, kernel_init=glorot_normal(), bias_init=constant(0.), rngs=rngs)
        #
        # self.log_std = nnx.Param(jnp.zeros(out_features))
        #
        # self.cri_dense1 = nnx.Linear(in_features, 64, kernel_init=glorot_normal(), bias_init=constant(0.), rngs=rngs)
        # self.cri_dense2 = nnx.Linear(64, 64, kernel_init=glorot_normal(), bias_init=constant(0.), rngs=rngs)
        # self.cri_dense3 = nnx.Linear(64, 1, kernel_init=glorot_normal(), bias_init=constant(0.), rngs=rngs)

    def __call__(self, x):

        if self.normalize:
            x = self.norm_layer(x)

        actor_mean = x
        for layer in self.act_layers:
            actor_mean = layer(actor_mean)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std.value))

        critic = x
        for layer in self.cri_layers:
            critic = layer(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class RecurrentActorCritic(nnx.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, rngs,
                 num_sequences: int,
                 lstm_net_arch: Sequence[int]=None, lstm_act_net_arch: Sequence[int]=None, lstm_cri_net_arch: Sequence[int]=None, lstm_activation: str=None,
                 net_arch: Sequence[int]=None, act_net_arch: Sequence[int]=None, cri_net_arch: Sequence[int]=None,
                 add_logistic_to_actor: bool = False, normalize:bool=False, is_feature_normalizable: Sequence[bool] = None
                 ):

        if act_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            act_net_arch = net_arch
        if cri_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            cri_net_arch = net_arch

        if lstm_act_net_arch is None:
            if lstm_net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            lstm_act_net_arch = lstm_net_arch
        if lstm_cri_net_arch is None:
            if lstm_net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            lstm_cri_net_arch = lstm_net_arch

        self.normalize = normalize
        if self.normalize:
            self.norm_layer = RunningNorm(num_features=in_features, use_bias=False, use_scale=False, rngs=rngs)

        activation = self.activation_from_name(activation)

        if lstm_activation is None:
            lstm_activation = activation
        else:
            lstm_activation = self.activation_from_name(lstm_activation)


        self.num_sequences = num_sequences

        lstm_act_net_arch = [num_sequences] + list(lstm_act_net_arch)

        self.lstm_act_layers = []
        for i in range(len(lstm_act_net_arch) - 1):
            self.lstm_act_layers.append(nnx.OptimizedLSTMCell(lstm_act_net_arch[i], lstm_act_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))

        lstm_cri_net_arch = [num_sequences] + list(lstm_cri_net_arch)

        self.lstm_cri_layers = []
        for i in range(len(lstm_cri_net_arch) - 1):
            self.lstm_cri_layers.append(nnx.OptimizedLSTMCell(lstm_cri_net_arch[i], lstm_cri_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))

        num_non_sequences = in_features - num_sequences

        act_net_arch = list(act_net_arch)
        cri_net_arch = list(cri_net_arch)

        act_net_arch = [num_non_sequences + lstm_act_net_arch[-1]] + act_net_arch + [out_features]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))
        if add_logistic_to_actor:
            self.act_layers.append(nnx.sigmoid)

        self.log_std = nnx.Param(jnp.zeros(out_features))

        cri_net_arch = [num_non_sequences + lstm_cri_net_arch[-1]] + cri_net_arch + [1]

        self.cri_layers = []
        for i in range(len(cri_net_arch) - 2):
            self.cri_layers.append(nnx.Linear(cri_net_arch[i], cri_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.cri_layers.append(activation)
        self.cri_layers.append(nnx.Linear(cri_net_arch[-2], cri_net_arch[-1], kernel_init=orthogonal(1.), bias_init=constant(0.), rngs=rngs))

    def __call__(self, x, lstm_act_state, lstm_cri_state):

        x = self.normalize_input(x)

        lstm_act_state, act_output = self.apply_lstm_act(x, lstm_act_state)
        pi = self.apply_act_mlp(x, act_output)

        lstm_cri_state, cri_output = self.apply_lstm_cri(x, lstm_cri_state)
        critic = self.apply_cri_mlp(x, cri_output)

        return pi, critic, lstm_act_state, lstm_cri_state

    def normalize_input(self, x):
        if self.normalize:
            return self.norm_layer(x)
        else:
            return x

    def apply_lstm_act(self, x, lstm_act_prev_state):
        seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_act_layers)):
            val = self.lstm_act_layers[i](lstm_act_prev_state[i], inputs)
            state, output = val
            states = states + (state,)
            inputs = output
        return states, inputs

    def apply_lstm_cri(self, x, lstm_cri_prev_state):
        seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_cri_layers)):
            state, output = self.lstm_cri_layers[i](lstm_cri_prev_state[i], inputs)
            states = states + (state,)
            inputs = output
        return states, inputs

    def apply_lstms_to_sequence(self, x, is_start_of_episode, lstm_act_state, lstm_cri_state, init_states):
        seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)

        def lstms_step(lstm_states, data):
            obs, start = data
            lstm_act_state, lstm_cri_state = jax.lax.cond(start, lambda: init_states, lambda: lstm_states)
            states = ()
            input = obs
            for i in range(len(self.lstm_act_layers)):
                state, output = self.lstm_act_layers[i](lstm_act_state[i], input)
                states = states + (state,)
                input = output
            output_act = input
            lstm_act_state = states

            states = ()
            input = obs
            for i in range(len(self.lstm_cri_layers)):
                state, output = self.lstm_cri_layers[i](lstm_cri_state[i], input)
                states = states + (state,)
                input = output
            output_cri = input
            lstm_cri_state = states

            return (lstm_act_state, lstm_cri_state), (output_act, output_cri)

        states, outputs = jax.lax.scan(lstms_step, (lstm_act_state, lstm_cri_state), (seq, is_start_of_episode))

        return  states, outputs



    def apply_act_mlp(self, x, lstm_act_output):
        non_seq = jax.lax.slice_in_dim(x, self.num_sequences, x.shape[-1], axis=-1)
        actor_mean = jnp.concat([lstm_act_output, non_seq], axis=-1)

        for layer in self.act_layers:
            actor_mean = layer(actor_mean)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(jnp.asarray(self.log_std)))

        return pi

    def apply_cri_mlp(self, x, lstm_cri_output):
        non_seq = jax.lax.slice_in_dim(x, self.num_sequences, x.shape[-1], axis=-1)
        critic = jnp.concat([lstm_cri_output, non_seq], axis=-1)

        for layer in self.cri_layers:
            critic = layer(critic)

        return jnp.squeeze(critic, axis=-1)

    def get_initial_lstm_state(self):
        init_act_state = ()
        inp_dim = self.num_sequences
        for layer in self.lstm_act_layers:
            init_act_state += (layer.initialize_carry((inp_dim,)),)
            inp_dim = layer.hidden_features

        init_cri_state = ()
        inp_dim = self.num_sequences
        for layer in self.lstm_cri_layers:
            init_cri_state += (layer.initialize_carry((inp_dim,)),)
            inp_dim = layer.hidden_features

        return init_act_state, init_cri_state

    @classmethod
    def activation_from_name(cls, name:str):
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



class StackedActorCritic(ActorCritic):

    def __init__(self, num_networks:int, in_features: int, out_features: int, activation: str, rngs, net_arch: list, act_net_arch: list, cri_net_arch: list, add_logistic_to_actor: bool, normalize:bool, is_feature_normalizable: Sequence[bool] = None):

        self.num_networks = num_networks

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, None, None, None, 0, None, None, None, None, None, None))
        def vmapped_fn(self, in_features: int, out_features: int, activation: str, rngs, net_arch: list, act_net_arch: list, cri_net_arch: list, add_logistic_to_actor: bool, normalize:bool, is_feature_normalizable: Sequence[bool]):
            super(StackedActorCritic, self).__init__(in_features, out_features, activation, rngs, net_arch=net_arch, act_net_arch=act_net_arch, cri_net_arch=cri_net_arch, add_logistic_to_actor=add_logistic_to_actor, normalize=normalize, is_feature_normalizable=is_feature_normalizable)

        vmapped_fn(self, in_features, out_features, activation, rngs, net_arch, act_net_arch, cri_net_arch, add_logistic_to_actor, normalize, is_feature_normalizable)

    def __call__(self, x):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, x):
            # distrax does not support vmap well
            # specifically distrax.MultivariateNormalDiag is cited in the README to give some problems
            # so before leaving vmap I return the arrays needed to reconstruct the distribution outside
            pi, val = super(StackedActorCritic, self).__call__(x)
            return pi.loc, pi.scale_diag, val

        pi_loc, pi_scale, val = vmapped_fn(self, x)
        pi = distrax.MultivariateNormalDiag(pi_loc, pi_scale)

        # jax.debug.print('shape SAC: {x}', x=x.shape)
        return pi, val

class StackedRecurrentActorCritic(RecurrentActorCritic):

    def __init__(self, num_networks:int, in_features: int, out_features: int, activation: str, rngs,
                 num_sequences: int,
                 lstm_net_arch: Sequence[int], lstm_act_net_arch: Sequence[int],
                 lstm_cri_net_arch: Sequence[int], lstm_activation: str,
                 net_arch: Sequence[int], act_net_arch: Sequence[int], cri_net_arch: Sequence[int],
                 add_logistic_to_actor: bool,
                 normalize:bool,
                 is_feature_normalizable: Sequence[bool] = None):

        self.num_networks = num_networks

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, None, None, None, 0, None, None, None, None, None, None, None, None, None, None, None))
        def vmapped_fn(self, in_features: int, out_features: int, activation: str, rngs,
                 num_sequences: int,
                 lstm_net_arch: Sequence[int], lstm_act_net_arch: Sequence[int],
                 lstm_cri_net_arch: Sequence[int], lstm_activation: str,
                 net_arch: Sequence[int], act_net_arch: Sequence[int], cri_net_arch: Sequence[int],
                 add_logistic_to_actor: bool,
                 normalize:bool,
                 is_feature_normalizable:Sequence[bool]):

            super(StackedRecurrentActorCritic, self).__init__(in_features, out_features, activation, rngs,
                     num_sequences,
                     lstm_net_arch=lstm_net_arch, lstm_act_net_arch=lstm_act_net_arch,
                     lstm_cri_net_arch=lstm_cri_net_arch, lstm_activation=lstm_activation,
                     net_arch=net_arch, act_net_arch=act_net_arch, cri_net_arch=cri_net_arch,
                     add_logistic_to_actor=add_logistic_to_actor,
                     normalize=normalize, is_feature_normalizable=is_feature_normalizable)

        vmapped_fn(self, in_features, out_features, activation, rngs, num_sequences, lstm_net_arch, lstm_act_net_arch,
                   lstm_cri_net_arch, lstm_activation, net_arch, act_net_arch, cri_net_arch, add_logistic_to_actor, normalize, is_feature_normalizable)

    def __call__(self, x, lstm_act_state, lstm_cri_state):

        x = self.normalize_input(x)

        lstm_act_state, act_output = self.apply_lstm_act(x, lstm_act_state)
        pi = self.apply_act_mlp(x, act_output)

        lstm_cri_state, cri_output = self.apply_lstm_cri(x, lstm_cri_state)
        critic = self.apply_cri_mlp(x, cri_output)

        return pi, critic, lstm_act_state, lstm_cri_state

    def normalize_input(self, x):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, x):
            return super(StackedRecurrentActorCritic, self).normalize_input(x)

        return vmapped_fn(self, x)

    def apply_lstm_act(self, x, lstm_act_prev_state):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, x, lstm_act_prev_state):
            return super(StackedRecurrentActorCritic, self).apply_lstm_act(x, lstm_act_prev_state)

        return vmapped_fn(self, x, lstm_act_prev_state)


    def apply_lstm_cri(self, x, lstm_cri_prev_state):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, x, lstm_cri_prev_state):
            return super(StackedRecurrentActorCritic, self).apply_lstm_cri(x, lstm_cri_prev_state)

        return vmapped_fn(self, x, lstm_cri_prev_state)

    def apply_act_mlp(self, x, lstm_act_output):

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, x, lstm_act_output):
            pi = super(StackedRecurrentActorCritic, self).apply_act_mlp(x, lstm_act_output)
            return pi.loc, pi.scale_diag

        pi_loc, pi_scale_diag = vmapped_fn(self, x, lstm_act_output)
        pi = distrax.MultivariateNormalDiag(pi_loc, pi_scale_diag)
        return pi

    def apply_cri_mlp(self, x, lstm_cri_output):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, x, lstm_cri_output):
            return super(StackedRecurrentActorCritic, self).apply_cri_mlp(x, lstm_cri_output)

        return vmapped_fn(self, x, lstm_cri_output)

    def get_initial_lstm_state(self):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self):
            return super(StackedRecurrentActorCritic, self).get_initial_lstm_state()

        return vmapped_fn(self)


class RECActorCritic(nnx.Module):
    def __init__(self, obs_keys:tuple, obs_is_local:dict, num_battery_agents: int, activation: str, rngs, net_arch: list=None, act_net_arch: list=None, cri_net_arch: list=None, passive_houses: bool=False, normalize:bool=False, is_obs_normalizable: Sequence[bool] = None):

        if act_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            act_net_arch = net_arch
        if cri_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            cri_net_arch = net_arch

        self.obs_keys = obs_keys
        self.obs_is_local = obs_is_local
        self.num_battery_agents = num_battery_agents

        in_features = len(obs_keys)

        self.normalize = normalize

        self.normalize = normalize
        if self.normalize:
            print('norm rec')
            self.norm_layer = RunningNorm(num_features=in_features, use_bias=False, use_scale=False, rngs=rngs)

        self.passive_houses = passive_houses

        act_net_arch = list(act_net_arch)
        cri_net_arch = list(cri_net_arch)

        activation = utils.activation_from_name(activation)

        act_net_arch = [in_features] + act_net_arch + [1]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)),
                                              bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(
            nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.),
                       rngs=rngs))

        cri_net_arch = [in_features] + cri_net_arch + [1]

        self.cri_layers = []
        for i in range(len(cri_net_arch) - 1):
            self.cri_layers.append(nnx.Linear(cri_net_arch[i], cri_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)),
                                              bias_init=constant(0.), rngs=rngs))
            self.cri_layers.append(activation)

        self.cri_layers.append(partial(jnp.squeeze, axis=-1))
        self.cri_layers.append(
            nnx.Linear(num_battery_agents, 1, kernel_init=orthogonal(1.), bias_init=constant(0.),
                       rngs=rngs))

    def __call__(self, obs):
        data = self.prepare_data(obs)

        # jax.debug.print('Shape Data: {x}', x=data.shape)

        logit = data
        for layer in self.act_layers:
            logit = layer(logit)

        alpha = nnx.softplus(logit).squeeze(axis=-1) + 1e-3
        # alpha = 10000. * nnx.sigmoid(logit).squeeze(axis=-1) + 1e-3
        # alpha = jnp.clip(alpha, max=1e+4)

        # jax.debug.print('alpha {x}', x=alpha, ordered=True)

        pi = distrax.Dirichlet(alpha)

        critic = data
        for layer in self.cri_layers:
            critic = layer(critic)

        # jax.debug.print('critic {x}', x=critic, ordered=True)

        return pi, jnp.squeeze(critic, axis=-1)

    def prepare_mask(self, is_obs_normalizable):
        mask = [is_obs_normalizable[key] for key in self.obs_keys if not self.obs_is_local[key]]
        mask += [is_obs_normalizable[key] for key in self.obs_keys if self.obs_is_local[key]]
        return mask

    def prepare_data(self, obs):

        local_obs = [obs[key] for key in self.obs_keys if self.obs_is_local[key]]
        global_obs = [obs[key] for key in self.obs_keys if not self.obs_is_local[key]]

        local_data = jnp.stack(local_obs, axis=-1)
        global_data = jnp.stack(global_obs, axis=-1)
        global_data = jnp.expand_dims(global_data, -2)
        global_data = jnp.repeat(global_data, self.num_battery_agents, axis=-2)

        data = jnp.concatenate((global_data, local_data), axis=-1)

        if self.normalize:
            data = self.norm_layer(data)

        return data


class RECRecurrentActorCritic(nnx.Module):
    def __init__(self, obs_keys: tuple,
                 obs_is_local: dict, obs_is_seq:dict,
                 num_battery_agents: int,
                 activation: str, rngs,
                 lstm_net_arch: Sequence[int] = None, lstm_act_net_arch: Sequence[int] = None,
                 lstm_cri_net_arch: Sequence[int] = None, lstm_activation: str = None,
                 net_arch: list=None, act_net_arch: list=None, cri_net_arch: list=None,
                 passive_houses: bool=False, normalize:bool=False, is_obs_normalizable: Sequence[bool] = None):

        if act_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            act_net_arch = net_arch
        if cri_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            cri_net_arch = net_arch

        if lstm_act_net_arch is None:
            if lstm_net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            lstm_act_net_arch = lstm_net_arch
        if lstm_cri_net_arch is None:
            if lstm_net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            lstm_cri_net_arch = lstm_net_arch

        self.obs_keys = obs_keys
        self.obs_is_local = obs_is_local
        self.obs_is_seq = obs_is_seq

        self.num_battery_agents = num_battery_agents
        self.passive_houses = passive_houses

        in_features = len(obs_keys)
        num_sequences = len([key for key in obs_keys if obs_is_seq[key]])

        self.num_sequences = num_sequences

        self.normalize = normalize
        if self.normalize:
            self.norm_layer = RunningNorm(num_features=in_features, use_bias=False, use_scale=False, rngs=rngs)

        activation = utils.activation_from_name(activation)

        if lstm_activation is None:
            lstm_activation = activation
        else:
            lstm_activation = utils.activation_from_name(lstm_activation)

        lstm_act_net_arch = [num_sequences] + list(lstm_act_net_arch)

        self.lstm_act_layers = []
        for i in range(len(lstm_act_net_arch) - 1):
            self.lstm_act_layers.append(nnx.OptimizedLSTMCell(lstm_act_net_arch[i], lstm_act_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))

        lstm_cri_net_arch = [num_sequences] + list(lstm_cri_net_arch)

        self.lstm_cri_layers = []
        for i in range(len(lstm_cri_net_arch) - 1):
            self.lstm_cri_layers.append(nnx.OptimizedLSTMCell(lstm_cri_net_arch[i], lstm_cri_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))

        num_non_sequences = in_features - num_sequences


        act_net_arch = list(act_net_arch)
        cri_net_arch = list(cri_net_arch)

        act_net_arch = [num_non_sequences + lstm_act_net_arch[-1]] + act_net_arch + [1]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))

        cri_net_arch = [num_non_sequences + lstm_cri_net_arch[-1]] + cri_net_arch + [1]

        self.cri_layers = []
        for i in range(len(cri_net_arch) - 1):
            self.cri_layers.append(nnx.Linear(cri_net_arch[i], cri_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.cri_layers.append(activation)

        self.cri_layers.append(partial(jnp.squeeze, axis=-1))
        self.cri_layers.append(nnx.Linear(num_battery_agents, 1, kernel_init=orthogonal(1.), bias_init=constant(0.), rngs=rngs))

    def __call__(self, obs, lstm_act_state, lstm_cri_state):
        data = self.prepare_data(obs)

        lstm_act_state, act_output = self.apply_lstm_act(data, lstm_act_state)
        pi = self.apply_act_mlp(data, act_output)

        lstm_cri_state, cri_output = self.apply_lstm_cri(data, lstm_cri_state)
        critic = self.apply_cri_mlp(data, cri_output)

        return pi, critic, lstm_act_state, lstm_cri_state

    def apply_lstm_act(self, x, lstm_act_prev_state):
        seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_act_layers)):
            state, output = self.lstm_act_layers[i](lstm_act_prev_state[i], inputs)
            states = states + (state,)
            inputs = output
        return states, inputs

    def apply_lstm_cri(self, x, lstm_cri_prev_state):
        seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_cri_layers)):
            state, output = self.lstm_cri_layers[i](lstm_cri_prev_state[i], inputs)
            states = states + (state,)
            inputs = output
        return states, inputs

    def apply_act_mlp(self, x, lstm_act_output):
        non_seq = jax.lax.slice_in_dim(x, self.num_sequences, x.shape[-1], axis=-1)
        logit = jnp.concat([lstm_act_output, non_seq], axis=-1)

        for layer in self.act_layers:
            logit = layer(logit)

        alpha = nnx.softplus(logit).squeeze(axis=-1) + 1e-3

        pi = distrax.Dirichlet(alpha)

        return pi

    def apply_cri_mlp(self, x, lstm_cri_output):
        non_seq = jax.lax.slice_in_dim(x, self.num_sequences, x.shape[-1], axis=-1)
        critic = jnp.concat([lstm_cri_output, non_seq], axis=-1)

        for layer in self.cri_layers:
            critic = layer(critic)

        return jnp.squeeze(critic, axis=-1)

    def prepare_mask(self, is_obs_normalizable):
        local_obs_keys = [key for key in self.obs_keys if self.obs_is_local[key]]
        global_obs_keys = [key for key in self.obs_keys if not self.obs_is_local[key]]

        mask = [is_obs_normalizable[key] for key in local_obs_keys if self.obs_is_seq[key]]
        mask += [is_obs_normalizable[key] for key in global_obs_keys if self.obs_is_seq[key]]

        mask += [is_obs_normalizable[key] for key in local_obs_keys if not self.obs_is_seq[key]]
        mask += [is_obs_normalizable[key] for key in global_obs_keys if not self.obs_is_seq[key]]

        return mask

    def prepare_data(self, obs):

        local_obs_keys = [key for key in self.obs_keys if self.obs_is_local[key]]
        global_obs_keys = [key for key in self.obs_keys if not self.obs_is_local[key]]

        seq_local_obs =  [obs[key] for key in local_obs_keys if self.obs_is_seq[key]]
        non_seq_local_obs = [obs[key] for key in local_obs_keys if not self.obs_is_seq[key]]

        seq_global_obs = [obs[key] for key in global_obs_keys if self.obs_is_seq[key]]
        non_seq_global_obs = [obs[key] for key in global_obs_keys if not self.obs_is_seq[key]]

        data_list = []

        if len(seq_local_obs) > 0:
            seq_local_data = jnp.stack(seq_local_obs, axis=-1)
            data_list.append(seq_local_data)
        if len(seq_global_obs) > 0:
            seq_global_data = jnp.stack(seq_global_obs, axis=-1)
            seq_global_data = jnp.expand_dims(seq_global_data, -2)
            seq_global_data = jnp.repeat(seq_global_data, self.num_battery_agents, axis=-2)
            data_list.append(seq_global_data)

        if len(non_seq_local_obs) > 0:
            non_seq_local_data = jnp.stack(non_seq_local_obs, axis=-1)
            data_list.append(non_seq_local_data)
        if len(non_seq_global_obs) > 0:
            non_seq_global_data = jnp.stack(non_seq_global_obs, axis=-1)
            non_seq_global_data = jnp.expand_dims(non_seq_global_data, -2)
            non_seq_global_data = jnp.repeat(non_seq_global_data, self.num_battery_agents, axis=-2)
            data_list.append(non_seq_global_data)

        data = jnp.concatenate(data_list, axis=-1)

        if self.normalize:
            data = self.norm_layer(data)

        return data

    def get_initial_lstm_state(self):
        init_act_state = ()
        inp_dim = self.num_sequences
        for layer in self.lstm_act_layers:
            init_act_state += (layer.initialize_carry((self.num_battery_agents, inp_dim)),)
            inp_dim = layer.hidden_features

        init_cri_state = ()
        inp_dim = self.num_sequences
        for layer in self.lstm_cri_layers:
            init_cri_state += (layer.initialize_carry((self.num_battery_agents, inp_dim)),)
            inp_dim = layer.hidden_features

        return init_act_state, init_cri_state

