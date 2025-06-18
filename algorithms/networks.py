import jax
import jax.numpy as jnp
from flax import nnx

from flax.nnx.nn.initializers import constant, orthogonal
import numpy as np
from typing import Sequence
import distrax

from algorithms.normalization_custom import RunningNorm


class ActorCritic(nnx.Module):
    def __init__(self,
                 obs_keys: tuple,
                 out_features: int,
                 activation: str,
                 rngs,
                 obs_keys_cri:tuple=None,
                 net_arch: list=None,
                 act_net_arch: list=None,
                 cri_net_arch: list=None,
                 add_logistic_to_actor: bool = False,
                 normalize: bool = True,
                 is_feature_normalizable:dict=None):

        if act_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            act_net_arch = net_arch
        if cri_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            cri_net_arch = net_arch

        self.obs_keys_act = obs_keys

        if obs_keys_cri is None:
            self.obs_keys_cri = obs_keys
        else:
            self.obs_keys_cri = obs_keys_cri


        print('ActorCritic', self.obs_keys_act, self.obs_keys_cri)

        act_net_arch = list(act_net_arch)
        cri_net_arch = list(cri_net_arch)

        self.normalize = normalize
        if normalize:
            print('norm batt')
            self.norm_layer_act = RunningNorm(num_features=len(self.obs_keys_act), use_bias=False, use_scale=False, rngs=rngs)
            self.norm_layer_cri = RunningNorm(num_features=len(self.obs_keys_cri), use_bias=False, use_scale=False, rngs=rngs)

        activation = activation_from_name(activation)

        act_net_arch = [len(self.obs_keys_act)] + act_net_arch + [out_features]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.1), bias_init=constant(0.), rngs=rngs))
        if add_logistic_to_actor:
            self.act_layers.append(nnx.sigmoid)

        self.log_std = nnx.Param(jnp.zeros(out_features))# - 1.)

        cri_net_arch = [len(self.obs_keys_cri)] + cri_net_arch + [1]

        self.cri_layers = []
        for i in range(len(cri_net_arch) - 2):
            self.cri_layers.append(nnx.Linear(cri_net_arch[i], cri_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.cri_layers.append(activation)
        self.cri_layers.append(nnx.Linear(cri_net_arch[-2], cri_net_arch[-1], kernel_init=orthogonal(1.), bias_init=constant(0.), rngs=rngs))

    def __call__(self, obs, return_cri=True):

        data_act, data_cri = self._prepare_data(obs, return_cri=return_cri)

        actor_mean = data_act
        for layer in self.act_layers:
            actor_mean = layer(actor_mean)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std.value))

        if return_cri:
            critic = data_cri
            for layer in self.cri_layers:
                critic = layer(critic)
            critic = jnp.squeeze(critic, axis=-1)
        else:
            critic = None

        return pi, critic

    def _prepare_data(self, obs, return_cri=True):
        data_act = jnp.stack([obs[key] for key in self.obs_keys_act], axis=-1)

        print('act', data_act.shape)

        if self.normalize:
            data_act = self.norm_layer_act(data_act)

        if return_cri:
            data_cri = jnp.stack([obs[key] for key in self.obs_keys_cri], axis=-1)
            print('cri', data_cri.shape)
            if self.normalize:
                data_cri = self.norm_layer_cri(data_cri)
        else:
            data_cri = None

        return data_act, data_cri

class RecurrentActorCritic(nnx.Module):
    def __init__(self,
                 obs_keys: tuple,
                 out_features: int,
                 activation: str,
                 rngs,
                 obs_is_seq:dict,
                 obs_keys_cri: tuple = None,
                 lstm_net_arch: Sequence[int]=None,
                 lstm_act_net_arch: Sequence[int]=None,
                 lstm_cri_net_arch: Sequence[int]=None,
                 lstm_activation: str=None,
                 net_arch: Sequence[int]=None,
                 act_net_arch: Sequence[int]=None,
                 cri_net_arch: Sequence[int]=None,
                 add_logistic_to_actor: bool = False,
                 normalize:bool=True,
                 is_feature_normalizable: Sequence[bool] = None
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

        self.obs_keys_act = obs_keys

        if obs_keys_cri is None:
            self.obs_keys_cri = obs_keys
        else:
            self.obs_keys_cri = obs_keys_cri

        self.normalize = normalize
        if self.normalize:
            self.norm_layer_act = RunningNorm(num_features=len(self.obs_keys_act), use_bias=False, use_scale=False, rngs=rngs)
            self.norm_layer_cri = RunningNorm(num_features=len(self.obs_keys_cri), use_bias=False, use_scale=False, rngs=rngs)

        activation = activation_from_name(activation)

        if lstm_activation is None:
            lstm_activation = activation
        else:
            lstm_activation = activation_from_name(lstm_activation)

        self.obs_is_seq = obs_is_seq
        self.num_sequences_act = len([key for key in self.obs_keys_act if self.obs_is_seq[key]])
        self.num_sequences_cri = len([key for key in self.obs_keys_cri if self.obs_is_seq[key]])

        lstm_act_net_arch = [self.num_sequences_act] + list(lstm_act_net_arch)

        self.lstm_act_layers = []
        for i in range(len(lstm_act_net_arch) - 1):
            self.lstm_act_layers.append(nnx.OptimizedLSTMCell(lstm_act_net_arch[i], lstm_act_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))

        lstm_cri_net_arch = [self.num_sequences_cri] + list(lstm_cri_net_arch)

        self.lstm_cri_layers = []
        for i in range(len(lstm_cri_net_arch) - 1):
            self.lstm_cri_layers.append(nnx.OptimizedLSTMCell(lstm_cri_net_arch[i], lstm_cri_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))

        num_non_sequences_arc = len(self.obs_keys_act) - self.num_sequences_act

        act_net_arch = list(act_net_arch)
        cri_net_arch = list(cri_net_arch)

        act_net_arch = [num_non_sequences_arc + lstm_act_net_arch[-1]] + act_net_arch + [out_features]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))
        if add_logistic_to_actor:
            self.act_layers.append(nnx.sigmoid)

        self.log_std = nnx.Param(jnp.zeros(out_features))

        num_non_sequences_cri = len(self.obs_keys_cri) - self.num_sequences_cri

        cri_net_arch = [num_non_sequences_cri + lstm_cri_net_arch[-1]] + cri_net_arch + [1]

        self.cri_layers = []
        for i in range(len(cri_net_arch) - 2):
            self.cri_layers.append(nnx.Linear(cri_net_arch[i], cri_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.cri_layers.append(activation)
        self.cri_layers.append(nnx.Linear(cri_net_arch[-2], cri_net_arch[-1], kernel_init=orthogonal(1.), bias_init=constant(0.), rngs=rngs))

    def __call__(self, obs, lstm_act_state, lstm_cri_state, return_cri=True):

        data_act, data_cri = self.prepare_data(obs, return_cri=return_cri)

        lstm_act_state, act_output = self.apply_lstm_act(data_act, lstm_act_state)
        pi = self.apply_act_mlp(data_act, act_output)

        if return_cri:
            lstm_cri_state, cri_output = self.apply_lstm_cri(data_cri, lstm_cri_state)
            critic = self.apply_cri_mlp(data_cri, cri_output)
        else:
            critic = None

        return pi, critic, lstm_act_state, lstm_cri_state

    def apply_lstm_act(self, data, lstm_act_prev_state):
        seq = jax.lax.slice_in_dim(data, 0, self.num_sequences_act, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_act_layers)):
            val = self.lstm_act_layers[i](lstm_act_prev_state[i], inputs)
            state, output = val
            states = states + (state,)
            inputs = output
        return states, inputs

    def apply_lstm_cri(self, data, lstm_cri_prev_state):
        seq = jax.lax.slice_in_dim(data, 0, self.num_sequences_cri, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_cri_layers)):
            state, output = self.lstm_cri_layers[i](lstm_cri_prev_state[i], inputs)
            states = states + (state,)
            inputs = output
        return states, inputs

    def apply_act_mlp(self, data, lstm_act_output):
        non_seq = jax.lax.slice_in_dim(data, self.num_sequences_act, data.shape[-1], axis=-1)
        actor_mean = jnp.concat([lstm_act_output, non_seq], axis=-1)

        for layer in self.act_layers:
            actor_mean = layer(actor_mean)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(jnp.asarray(self.log_std)))

        return pi

    def apply_cri_mlp(self, data, lstm_cri_output):
        non_seq = jax.lax.slice_in_dim(data, self.num_sequences_cri, data.shape[-1], axis=-1)
        critic = jnp.concat([lstm_cri_output, non_seq], axis=-1)

        for layer in self.cri_layers:
            critic = layer(critic)

        return jnp.squeeze(critic, axis=-1)

    def get_initial_lstm_state(self):
        init_act_state = ()
        inp_dim = self.num_sequences_act
        for layer in self.lstm_act_layers:
            init_act_state += (layer.initialize_carry((inp_dim,)),)
            inp_dim = layer.hidden_features

        init_cri_state = ()
        inp_dim = self.num_sequences_cri
        for layer in self.lstm_cri_layers:
            init_cri_state += (layer.initialize_carry((inp_dim,)),)
            inp_dim = layer.hidden_features

        return init_act_state, init_cri_state

    def prepare_data(self, obs, return_cri=True):
        def _prepare_data(obs_keys):
            seq_obs = [obs[key] for key in obs_keys if self.obs_is_seq[key]]
            non_seq_obs = [obs[key] for key in obs_keys if not self.obs_is_seq[key]]

            data_list = seq_obs + non_seq_obs

            data = jnp.stack(data_list, axis=-1)
            return data

        data_act = _prepare_data(self.obs_keys_act)
        if self.normalize:
            data_act = self.norm_layer_act(data_act)

        if return_cri:
            data_cri = _prepare_data(self.obs_keys_cri)
            if self.normalize:
                data_cri = self.norm_layer_cri(data_cri)
        else:
            data_cri = None

        return data_act, data_cri


class StackedActorCritic(ActorCritic):

    def __init__(self,
                 num_networks:int,
                 obs_keys: tuple,
                 out_features: int,
                 activation: str,
                 rngs,
                 obs_keys_cri: tuple = None,
                 net_arch: list = None,
                 act_net_arch: list = None,
                 cri_net_arch: list = None,
                 add_logistic_to_actor: bool = False,
                 normalize: bool = True,
                 is_feature_normalizable:dict = None):

        self.num_networks = num_networks

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, None, None, None, 0, None, None, None, None, None, None, None))
        def vmapped_fn(self, obs_keys:tuple, out_features: int, activation: str, rngs, obs_keys_cri:tuple, net_arch: list, act_net_arch: list, cri_net_arch: list, add_logistic_to_actor: bool, normalize:bool, is_feature_normalizable:dict):
            super(StackedActorCritic, self).__init__(obs_keys, out_features, activation, rngs, obs_keys_cri=obs_keys_cri, net_arch=net_arch, act_net_arch=act_net_arch, cri_net_arch=cri_net_arch, add_logistic_to_actor=add_logistic_to_actor, normalize=normalize, is_feature_normalizable=is_feature_normalizable)

        vmapped_fn(self, obs_keys, out_features, activation, rngs, obs_keys_cri, net_arch, act_net_arch, cri_net_arch, add_logistic_to_actor, normalize, is_feature_normalizable)

    def __call__(self, obs, return_cri=True):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, 0, None))
        def vmapped_fn(self, obs, return_cri):
            # distrax does not support vmap well
            # specifically distrax.MultivariateNormalDiag is cited in the README to give some problems
            # so before leaving vmap I return the arrays needed to reconstruct the distribution outside
            pi, val = super(StackedActorCritic, self).__call__(obs, return_cri=return_cri)
            return pi.loc, pi.scale_diag, val

        pi_loc, pi_scale, val = vmapped_fn(self, obs, return_cri=return_cri)
        pi = distrax.MultivariateNormalDiag(pi_loc, pi_scale)

        return pi, val

class StackedRecurrentActorCritic(RecurrentActorCritic):

    def __init__(self,
                 num_networks:int,
                 obs_keys: tuple,
                 out_features: int,
                 activation: str,
                 rngs,
                 obs_is_seq:dict,
                 obs_keys_cri: tuple = None,
                 lstm_net_arch: Sequence[int]=None,
                 lstm_act_net_arch: Sequence[int]=None,
                 lstm_cri_net_arch: Sequence[int]=None,
                 lstm_activation: str=None,
                 net_arch: Sequence[int]=None,
                 act_net_arch: Sequence[int]=None,
                 cri_net_arch: Sequence[int]=None,
                 add_logistic_to_actor: bool=False,
                 normalize:bool=True,
                 is_feature_normalizable: Sequence[bool] = None):

        self.num_networks = num_networks

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, None, None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None))
        def vmapped_fn(self,
                       obs_keys,
                       out_features: int,
                       activation: str,
                       rngs,
                       obs_is_seq: dict,
                       obs_keys_cri: tuple,
                       lstm_net_arch: Sequence[int], lstm_act_net_arch: Sequence[int],
                       lstm_cri_net_arch: Sequence[int], lstm_activation: str,
                       net_arch: Sequence[int], act_net_arch: Sequence[int], cri_net_arch: Sequence[int],
                       add_logistic_to_actor: bool,
                       normalize:bool,
                       is_feature_normalizable:Sequence[bool]):

            super(StackedRecurrentActorCritic, self).__init__(obs_keys, out_features, activation, rngs,
                     obs_is_seq, obs_keys_cri,
                     lstm_net_arch=lstm_net_arch, lstm_act_net_arch=lstm_act_net_arch,
                     lstm_cri_net_arch=lstm_cri_net_arch, lstm_activation=lstm_activation,
                     net_arch=net_arch, act_net_arch=act_net_arch, cri_net_arch=cri_net_arch,
                     add_logistic_to_actor=add_logistic_to_actor,
                     normalize=normalize, is_feature_normalizable=is_feature_normalizable)

        vmapped_fn(self, obs_keys, out_features, activation, rngs, obs_is_seq, obs_keys_cri, lstm_net_arch, lstm_act_net_arch,
                   lstm_cri_net_arch, lstm_activation, net_arch, act_net_arch, cri_net_arch, add_logistic_to_actor, normalize, is_feature_normalizable)

    def __call__(self, obs, lstm_act_state, lstm_cri_state, return_cri=True):

        data_act, data_cri = self.prepare_data(obs, return_cri=return_cri)

        lstm_act_state, act_output = self.apply_lstm_act(data_act, lstm_act_state)
        pi = self.apply_act_mlp(data_act, act_output)

        if return_cri:
            lstm_cri_state, cri_output = self.apply_lstm_cri(data_cri, lstm_cri_state)
            critic = self.apply_cri_mlp(data_cri, cri_output)
        else:
            critic = None

        return pi, critic, lstm_act_state, lstm_cri_state

    def apply_lstm_act(self, data, lstm_act_prev_state):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, data, lstm_act_prev_state):
            return super(StackedRecurrentActorCritic, self).apply_lstm_act(data, lstm_act_prev_state)

        return vmapped_fn(self, data, lstm_act_prev_state)


    def apply_lstm_cri(self, data, lstm_cri_prev_state):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, data, lstm_cri_prev_state):
            return super(StackedRecurrentActorCritic, self).apply_lstm_cri(data, lstm_cri_prev_state)

        return vmapped_fn(self, data, lstm_cri_prev_state)

    def apply_act_mlp(self, data, lstm_act_output):

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, data, lstm_act_output):
            pi = super(StackedRecurrentActorCritic, self).apply_act_mlp(data, lstm_act_output)
            return pi.loc, pi.scale_diag

        pi_loc, pi_scale_diag = vmapped_fn(self, data, lstm_act_output)
        pi = distrax.MultivariateNormalDiag(pi_loc, pi_scale_diag)
        return pi

    def apply_cri_mlp(self, data, lstm_cri_output):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, data, lstm_cri_output):
            return super(StackedRecurrentActorCritic, self).apply_cri_mlp(data, lstm_cri_output)

        return vmapped_fn(self, data, lstm_cri_output)

    def get_initial_lstm_state(self):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self):
            return super(StackedRecurrentActorCritic, self).get_initial_lstm_state()

        return vmapped_fn(self)

    def prepare_data(self, obs, return_cri=True):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, 0, None))
        def vmapped_fun(self, obs, return_cri):
            return super(StackedRecurrentActorCritic, self).prepare_data(obs, return_cri=return_cri)

        return vmapped_fun(self, obs, return_cri)

class RECActorCritic(nnx.Module):
    def __init__(self,
                 obs_keys:tuple,
                 obs_is_local:dict,
                 num_battery_agents: int,
                 activation: str,
                 rngs,
                 obs_keys_cri: tuple=None,
                 net_arch: tuple=(),
                 act_net_arch: tuple=None,
                 cri_net_arch: tuple=None,
                 non_shared_net_arch_before: tuple=(),
                 non_shared_net_arch_after: tuple=(),
                 passive_houses: bool=False,
                 normalize:bool=True,
                 is_obs_normalizable: Sequence[bool] = None):

        if act_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            act_net_arch = net_arch
        if cri_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            cri_net_arch = net_arch

        self.obs_keys_act = obs_keys
        if obs_keys_cri is None:
            self.obs_keys_cri = obs_keys
        else:
            self.obs_keys_cri = obs_keys_cri

        print('RECActorCritic', self.obs_keys_act, self.obs_keys_cri)

        self.obs_is_local = obs_is_local
        self.num_battery_agents = num_battery_agents

        in_features_act = len(self.obs_keys_act)
        in_features_cri = len(self.obs_keys_cri)

        self.normalize = normalize

        self.normalize = normalize
        if self.normalize:
            print('norm rec')
            self.norm_layer_act = RunningNorm(num_features=in_features_act, use_bias=False, use_scale=False, rngs=rngs)
            self.norm_layer_cri = RunningNorm(num_features=in_features_cri, use_bias=False, use_scale=False, rngs=rngs)

        self.passive_houses = passive_houses

        act_net_arch = tuple(act_net_arch)
        cri_net_arch = tuple(cri_net_arch)

        activation = activation_from_name(activation)

        self.act_layers_before, self.act_layers, self.act_layers_after = build_layers_rec(in_features_act, non_shared_net_arch_before,
                                                                                          act_net_arch, non_shared_net_arch_after,
                                                                                          num_battery_agents, activation, rngs)
        self.cri_layers_before, self.cri_layers, self.cri_layers_after = build_layers_rec(in_features_cri, non_shared_net_arch_before,
                                                                                          cri_net_arch, non_shared_net_arch_after,
                                                                                          num_battery_agents, activation, rngs)

        self.cri_mixer = nnx.Param(jnp.zeros(self.num_battery_agents))

        self.activation = activation

    def __call__(self, obs, return_cri=True):
        data_act, data_cri = self.prepare_data(obs, return_cri=return_cri)

        print('dataaa act', data_act.shape)
        if return_cri:
            print('dataaa cri', data_cri.shape)

        # jax.debug.print('Shape Data: {x}', x=data.shape)

        logit = data_act

        logit = self.call_non_shared_layers(self.act_layers_before, logit)

        for layer in self.act_layers:
            logit = layer(logit)

        logit = self.call_non_shared_layers(self.act_layers_after, logit)
        logit = logit.squeeze(axis=-1)
        print('logit', logit.shape)

        alpha = nnx.softplus(logit) + 1e-3

        pi = distrax.Dirichlet(alpha)

        def finalize_critic(critic_separate):
            weighted = critic_separate * nnx.softmax(self.cri_mixer.value)
            return weighted.sum(axis=-1)

        if return_cri:
            critic = data_cri

            critic = self.call_non_shared_layers(self.cri_layers_before, critic)
            for layer in self.cri_layers:
                critic = layer(critic)

            critic = self.call_non_shared_layers(self.cri_layers_after, critic)
            critic =  jnp.squeeze(critic, axis=-1)
            critic = finalize_critic(critic)
        else:
            critic = None

        return pi, critic

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

    def prepare_data(self, obs, return_cri=True):

        local_obs = [obs[key] for key in self.obs_keys_act if self.obs_is_local[key]]
        global_obs = [obs[key] for key in self.obs_keys_act if not self.obs_is_local[key]]

        local_data = jnp.stack(local_obs, axis=-1)
        global_data = jnp.stack(global_obs, axis=-1)
        global_data = jnp.expand_dims(global_data, -2)
        global_data = jnp.repeat(global_data, self.num_battery_agents, axis=-2)

        data_act = jnp.concatenate((global_data, local_data), axis=-1)

        if self.normalize:
            data_act = self.norm_layer_act(data_act)

        if return_cri:
            local_obs = [obs[key] for key in self.obs_keys_cri if self.obs_is_local[key]]
            global_obs = [obs[key] for key in self.obs_keys_cri if not self.obs_is_local[key]]

            local_data = jnp.stack(local_obs, axis=-1)
            global_data = jnp.stack(global_obs, axis=-1)
            global_data = jnp.expand_dims(global_data, -2)
            global_data = jnp.repeat(global_data, self.num_battery_agents, axis=-2)

            data_cri = jnp.concatenate((global_data, local_data), axis=-1)

            if self.normalize:
                data_cri = self.norm_layer_cri(data_cri)
        else:
            data_cri = None

        return data_act, data_cri


class RECRecurrentActorCritic(nnx.Module):
    def __init__(self,
                 obs_keys: tuple,
                 obs_is_local: dict,
                 obs_is_seq:dict,
                 num_battery_agents: int,
                 activation: str,
                 rngs,
                 obs_keys_cri: tuple = None,
                 lstm_net_arch: Sequence[int] = None,
                 lstm_act_net_arch: Sequence[int] = None,
                 lstm_cri_net_arch: Sequence[int] = None,
                 lstm_activation: str = None,
                 net_arch: tuple=(),
                 act_net_arch: tuple=None,
                 cri_net_arch: tuple=None,
                 non_shared_net_arch_before: tuple=(),
                 non_shared_net_arch_after: tuple=(),
                 share_lstm_batteries=True,
                 passive_houses: bool=False,
                 normalize:bool=True,
                 is_obs_normalizable: Sequence[bool] = None):

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

        self.obs_keys_act = obs_keys
        if obs_keys_cri is None:
            self.obs_keys_cri = obs_keys
        else:
            self.obs_keys_cri = obs_keys_cri

        self.obs_is_local = obs_is_local
        self.obs_is_seq = obs_is_seq

        self.num_battery_agents = num_battery_agents
        self.passive_houses = passive_houses
        self.share_lstm_batteries = share_lstm_batteries

        self.num_sequences_act = len([key for key in self.obs_keys_act if self.obs_is_seq[key]])
        self.num_sequences_cri = len([key for key in self.obs_keys_cri if self.obs_is_seq[key]])

        in_features_act = len(self.obs_keys_act)
        in_features_cri = len(self.obs_keys_cri)

        self.normalize = normalize
        if self.normalize:
            self.norm_layer_act = RunningNorm(num_features=in_features_act, use_bias=False, use_scale=False, rngs=rngs)
            self.norm_layer_cri = RunningNorm(num_features=in_features_cri, use_bias=False, use_scale=False, rngs=rngs)

        activation = activation_from_name(activation)

        if lstm_activation is None:
            lstm_activation = activation
        else:
            lstm_activation = activation_from_name(lstm_activation)

        @nnx.split_rngs(splits=num_battery_agents)
        @nnx.vmap(in_axes=(None, None, None, 0))
        def stacked_lstm_layer(in_features, out_features, activation_fn, rngs):
            return nnx.OptimizedLSTMCell(in_features, out_features, activation_fn=activation_fn, rngs=rngs)

        lstm_act_net_arch = (self.num_sequences_act,) + tuple(lstm_act_net_arch)
        lstm_cri_net_arch = (self.num_sequences_cri,) + tuple(lstm_cri_net_arch)

        self.lstm_act_layers = []
        self.lstm_cri_layers = []

        if share_lstm_batteries:
            for i in range(len(lstm_act_net_arch) - 1):
                self.lstm_act_layers.append(nnx.OptimizedLSTMCell(lstm_act_net_arch[i], lstm_act_net_arch[i+1],
                                                                  activation_fn=lstm_activation, rngs=rngs))
            for i in range(len(lstm_cri_net_arch) - 1):
                self.lstm_cri_layers.append(nnx.OptimizedLSTMCell(lstm_cri_net_arch[i], lstm_cri_net_arch[i+1],
                                                                  activation_fn=lstm_activation, rngs=rngs))
        else:
            for i in range(len(lstm_act_net_arch) - 1):
                self.lstm_act_layers.append(stacked_lstm_layer(lstm_act_net_arch[i], lstm_act_net_arch[i+1],
                                                               activation_fn=lstm_activation, rngs=rngs))
            for i in range(len(lstm_cri_net_arch) - 1):
                self.lstm_cri_layers.append(stacked_lstm_layer(lstm_cri_net_arch[i], lstm_cri_net_arch[i+1],
                                                               activation_fn=lstm_activation, rngs=rngs))

        num_non_sequences_act = in_features_act - self.num_sequences_act
        num_non_sequences_cri = in_features_cri - self.num_sequences_cri

        self.act_layers_before, self.act_layers, self.act_layers_after = build_layers_rec(num_non_sequences_act + lstm_act_net_arch[-1],
                                                                                          non_shared_net_arch_before,
                                                                                          act_net_arch,
                                                                                          non_shared_net_arch_after,
                                                                                          num_battery_agents, activation, rngs)
        self.cri_layers_before, self.cri_layers, self.cri_layers_after = build_layers_rec(num_non_sequences_cri + lstm_cri_net_arch[-1],
                                                                                          non_shared_net_arch_before,
                                                                                          cri_net_arch,
                                                                                          non_shared_net_arch_after,
                                                                                          num_battery_agents, activation, rngs)

        self.cri_mixer = nnx.Param(jnp.zeros(self.num_battery_agents))

    def __call__(self, obs, lstm_act_state, lstm_cri_state, return_cri=True):
        data_act, data_cri = self.prepare_data(obs, return_cri=return_cri)

        lstm_act_state, act_output = self.apply_lstm_act(data_act, lstm_act_state)
        pi = self.apply_act_mlp(data_act, act_output)

        if return_cri:
            lstm_cri_state, cri_output = self.apply_lstm_cri(data_cri, lstm_cri_state)
            critic = self.apply_cri_mlp(data_cri, cri_output)
        else:
            critic = None

        return pi, critic, lstm_act_state, lstm_cri_state

    def apply_lstm_act(self, x, lstm_act_prev_state):

        @nnx.split_rngs(splits=self.num_battery_agents)
        @nnx.vmap(in_axes=(0, -2, -2), out_axes=(-2, -2))
        def compute_layer(lay, state, data):
            return lay(state, data)

        seq = jax.lax.slice_in_dim(x, 0, self.num_sequences_act, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_act_layers)):
            if self.share_lstm_batteries:
                state, output = self.lstm_act_layers[i](lstm_act_prev_state[i], inputs)
            else:
                state, output = compute_layer(self.lstm_act_layers[i], lstm_act_prev_state[i], inputs)
            states = states + (state,)
            inputs = output
        return states, inputs

    def apply_lstm_cri(self, x, lstm_cri_prev_state):

        @nnx.split_rngs(splits=self.num_battery_agents)
        @nnx.vmap(in_axes=(0, -2, -2), out_axes=(-2, -2))
        def compute_layer(lay, state, data):
            return lay(state, data)

        seq = jax.lax.slice_in_dim(x, 0, self.num_sequences_cri, axis=-1)
        states = ()
        inputs = seq
        for i in range(len(self.lstm_cri_layers)):
            if self.share_lstm_batteries:
                state, output = self.lstm_cri_layers[i](lstm_cri_prev_state[i], inputs)
            else:
                state, output = compute_layer(self.lstm_cri_layers[i], lstm_cri_prev_state[i], inputs)
            states = states + (state,)
            inputs = output
        return states, inputs

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

    def apply_act_mlp(self, x, lstm_act_output):
        non_seq = jax.lax.slice_in_dim(x, self.num_sequences_act, x.shape[-1], axis=-1)
        logit = jnp.concat([lstm_act_output, non_seq], axis=-1)

        logit = self.call_non_shared_layers(self.act_layers_before, logit)
        for layer in self.act_layers:
            logit = layer(logit)
        logit = self.call_non_shared_layers(self.act_layers_after, logit)

        alpha = nnx.softplus(logit).squeeze(axis=-1) + 1e-3

        pi = distrax.Dirichlet(alpha)

        return pi

    def apply_cri_mlp(self, x, lstm_cri_output):
        def finalize_critic(critic_separate):
            weighted = critic_separate * nnx.softmax(self.cri_mixer.value)
            return weighted.sum(axis=-1)

        non_seq = jax.lax.slice_in_dim(x, self.num_sequences_cri, x.shape[-1], axis=-1)
        critic = jnp.concat([lstm_cri_output, non_seq], axis=-1)

        critic = self.call_non_shared_layers(self.cri_layers_before, critic)
        for layer in self.cri_layers:
            critic = layer(critic)
        critic = self.call_non_shared_layers(self.cri_layers_after, critic)

        critic = jnp.squeeze(critic, axis=-1)

        critic = finalize_critic(critic)

        return critic

    def prepare_data(self, obs, return_cri=True):

        def _prepare_data(obs, obs_keys):

            local_obs_keys = [key for key in obs_keys if self.obs_is_local[key]]
            global_obs_keys = [key for key in obs_keys if not self.obs_is_local[key]]

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

            return data

        data_act = _prepare_data(obs, self.obs_keys_act)

        if self.normalize:
            data_act = self.norm_layer_act(data_act)

        if return_cri:
            data_cri = _prepare_data(obs, self.obs_keys_cri)
            if self.normalize:
                data_cri = self.norm_layer_cri(data_cri)
        else:
            data_cri = None

        return data_act, data_cri

    def get_initial_lstm_state(self):
        init_act_state = ()
        inp_dim = self.num_sequences_act
        for layer in self.lstm_act_layers:
            init_act_state += (layer.initialize_carry((self.num_battery_agents, inp_dim)),)
            inp_dim = layer.hidden_features

        init_cri_state = ()
        inp_dim = self.num_sequences_cri
        for layer in self.lstm_cri_layers:
            init_cri_state += (layer.initialize_carry((self.num_battery_agents, inp_dim)),)
            inp_dim = layer.hidden_features

        return init_act_state, init_cri_state


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

        self.layers_before, self.layers, self.layers_after = build_layers_rec(in_features, non_shared_net_arch_before,
                                                                              net_arch, non_shared_net_arch_after,
                                                                              num_battery_agents, activation, rngs)

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

def build_stacked_layer(num_battery_agents, in_features, out_features, kernel_init, bias_init, rngs):
    @nnx.split_rngs(splits=num_battery_agents)
    @nnx.vmap(in_axes=(None, None, None, None, 0))
    def _stacked_layer(in_features, out_features, kernel_init, bias_init, rngs):
        return nnx.Linear(in_features, out_features, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)

    return _stacked_layer(in_features, out_features, kernel_init, bias_init, rngs)


def build_layers_rec(input_size, net_arch_before, net_arch, net_arch_after, num_battery_agents, activation, rngs):

    layers_before = []

    last_len = input_size

    if len(net_arch_before) != 0:
        net_arch_before = (last_len,) + net_arch_before
        last_len = net_arch_before[-1]

        for i in range(len(net_arch_before) - 1):
            layers_before.append(build_stacked_layer(num_battery_agents, net_arch_before[i], net_arch_before[i+1],
                                                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            layers_before.append(activation)

        if len(net_arch) == 0 and len(net_arch_after) == 0:
            layers_before.append(build_stacked_layer(num_battery_agents, net_arch_before[-1], 1,
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
            layers_after.append(build_stacked_layer(num_battery_agents, net_arch_after[i], net_arch_after[i+1],
                                                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            layers_after.append(activation)
        layers_after.append(build_stacked_layer(num_battery_agents, net_arch_after[-1], 1,
                                                kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))

    return layers_before, layers, layers_after
