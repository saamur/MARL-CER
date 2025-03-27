import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from functools import partial

from jax_tqdm import scan_tqdm, loop_tqdm

from flax.nnx.nn.initializers import constant, orthogonal, glorot_normal
from flax.nnx import GraphDef, GraphState
import numpy as np
import optax
from typing import Sequence, NamedTuple, Any
import distrax

from algorithms.normalization_custom import RunningNorm
import algorithms.utils as utils
from ernestogym.envs_jax.single_agent.env import  MicroGridEnv

import algorithms.recurrent_custom as recurrent_custom

from .wrappers import (
    LogWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

# class RecurrentActorCritic(nnx.Module):
#     def __init__(self, in_features: int, out_features: int, activation: str, rngs,
#                  num_sequences: int,
#                  lstm_net_arch: Sequence[int]=None, lstm_act_net_arch: Sequence[int]=None, lstm_cri_net_arch: Sequence[int]=None, lstm_activation: str=None,
#                  net_arch: Sequence[int]=None, act_net_arch: Sequence[int]=None, cri_net_arch: Sequence[int]=None,
#                  add_logistic_to_actor: bool = False, normalize:bool=False, is_feature_normalizable: Sequence[bool] = None
#                  ):
#
#         if act_net_arch is None:
#             if net_arch is None:
#                 raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
#             act_net_arch = net_arch
#         if cri_net_arch is None:
#             if net_arch is None:
#                 raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
#             cri_net_arch = net_arch
#
#         if lstm_act_net_arch is None:
#             if lstm_net_arch is None:
#                 raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
#             lstm_act_net_arch = lstm_net_arch
#         if lstm_cri_net_arch is None:
#             if lstm_net_arch is None:
#                 raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
#             lstm_cri_net_arch = lstm_net_arch
#
#         self.normalize = normalize
#         if self.normalize:
#             self.norm_layer = RunningNorm(num_features=in_features, use_bias=False, use_scale=False, rngs=rngs)
#
#         activation = self.activation_from_name(activation)
#
#         if lstm_activation is None:
#             lstm_activation = activation
#         else:
#             lstm_activation = self.activation_from_name(lstm_activation)
#
#
#         self.num_sequences = num_sequences
#
#         lstm_act_net_arch = [num_sequences] + list(lstm_act_net_arch)
#
#         self.lstm_act_layers = []
#         for i in range(len(lstm_act_net_arch) - 1):
#             self.lstm_act_layers.append(nnx.OptimizedLSTMCell(lstm_act_net_arch[i], lstm_act_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))
#
#         lstm_cri_net_arch = [num_sequences] + list(lstm_cri_net_arch)
#
#         self.lstm_cri_layers = []
#         for i in range(len(lstm_cri_net_arch) - 1):
#             self.lstm_cri_layers.append(nnx.OptimizedLSTMCell(lstm_cri_net_arch[i], lstm_cri_net_arch[i+1], activation_fn=lstm_activation, rngs=rngs))
#
#         num_non_sequences = in_features - num_sequences
#
#         act_net_arch = list(act_net_arch)
#         cri_net_arch = list(cri_net_arch)
#
#         act_net_arch = [num_non_sequences + lstm_act_net_arch[-1]] + act_net_arch + [out_features]
#
#         self.act_layers = []
#         for i in range(len(act_net_arch) - 2):
#             self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
#             self.act_layers.append(activation)
#         self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))
#         if add_logistic_to_actor:
#             self.act_layers.append(nnx.sigmoid)
#
#         self.log_std = nnx.Param(jnp.zeros(out_features))
#
#         cri_net_arch = [num_non_sequences + lstm_cri_net_arch[-1]] + cri_net_arch + [1]
#
#         self.cri_layers = []
#         for i in range(len(cri_net_arch) - 2):
#             self.cri_layers.append(nnx.Linear(cri_net_arch[i], cri_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
#             self.cri_layers.append(activation)
#         self.cri_layers.append(nnx.Linear(cri_net_arch[-2], cri_net_arch[-1], kernel_init=orthogonal(1.), bias_init=constant(0.), rngs=rngs))
#
#     def __call__(self, x, lstm_act_state, lstm_cri_state):
#
#         x = self.normalize_input(x)
#
#         lstm_act_state, act_output = self.apply_lstm_act(x, lstm_act_state)
#         pi = self.apply_act_mlp(x, act_output)
#
#         lstm_cri_state, cri_output = self.apply_lstm_cri(x, lstm_cri_state)
#         critic = self.apply_cri_mlp(x, cri_output)
#
#         return pi, critic, lstm_act_state, lstm_cri_state
#
#     def normalize_input(self, x):
#         if self.normalize:
#             return self.norm_layer(x)
#         else:
#             return x
#
#     def apply_lstm_act(self, x, lstm_act_prev_state):
#         seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)
#         states = ()
#         inputs = seq
#         for i in range(len(self.lstm_act_layers)):
#             val = self.lstm_act_layers[i](lstm_act_prev_state[i], inputs)
#             state, output = val
#             states = states + (state,)
#             inputs = output
#         return states, inputs
#
#     def apply_lstm_cri(self, x, lstm_cri_prev_state):
#         seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)
#         states = ()
#         inputs = seq
#         for i in range(len(self.lstm_cri_layers)):
#             state, output = self.lstm_cri_layers[i](lstm_cri_prev_state[i], inputs)
#             states = states + (state,)
#             inputs = output
#         return states, inputs
#
#     def apply_lstms_to_sequence(self, x, is_start_of_episode, lstm_act_state, lstm_cri_state, init_states):
#         seq = jax.lax.slice_in_dim(x, 0, self.num_sequences, axis=-1)
#
#         def lstms_step(lstm_states, data):
#             obs, start = data
#             lstm_act_state, lstm_cri_state = jax.lax.cond(start, lambda: init_states, lambda: lstm_states)
#             states = ()
#             input = obs
#             for i in range(len(self.lstm_act_layers)):
#                 state, output = self.lstm_act_layers[i](lstm_act_state[i], input)
#                 states = states + (state,)
#                 input = output
#             output_act = input
#             lstm_act_state = states
#
#             states = ()
#             input = obs
#             for i in range(len(self.lstm_cri_layers)):
#                 state, output = self.lstm_cri_layers[i](lstm_cri_state[i], input)
#                 states = states + (state,)
#                 input = output
#             output_cri = input
#             lstm_cri_state = states
#
#             return (lstm_act_state, lstm_cri_state), (output_act, output_cri)
#
#         states, outputs = jax.lax.scan(lstms_step, (lstm_act_state, lstm_cri_state), (seq, is_start_of_episode))
#
#         return  states, outputs
#
#
#
#     def apply_act_mlp(self, x, lstm_act_output):
#         non_seq = jax.lax.slice_in_dim(x, self.num_sequences, x.shape[-1], axis=-1)
#         actor_mean = jnp.concat([lstm_act_output, non_seq], axis=-1)
#
#         for layer in self.act_layers:
#             actor_mean = layer(actor_mean)
#
#         pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(jnp.asarray(self.log_std)))
#
#         return pi
#
#     def apply_cri_mlp(self, x, lstm_cri_output):
#         non_seq = jax.lax.slice_in_dim(x, self.num_sequences, x.shape[-1], axis=-1)
#         critic = jnp.concat([lstm_cri_output, non_seq], axis=-1)
#
#         for layer in self.cri_layers:
#             critic = layer(critic)
#
#         return jnp.squeeze(critic, axis=-1)
#
#     def get_initial_lstm_state(self):
#         init_act_state = ()
#         inp_dim = self.num_sequences
#         for layer in self.lstm_act_layers:
#             init_act_state += (layer.initialize_carry((inp_dim,)),)
#             inp_dim = layer.hidden_features
#
#         init_cri_state = ()
#         inp_dim = self.num_sequences
#         for layer in self.lstm_cri_layers:
#             init_cri_state += (layer.initialize_carry((inp_dim,)),)
#             inp_dim = layer.hidden_features
#
#         return init_act_state, init_cri_state
#
#     @classmethod
#     def activation_from_name(cls, name:str):
#         name = name.lower()
#         if name == 'relu':
#             return nnx.relu
#         elif name == 'tanh':
#             return nnx.tanh
#         elif name == 'sigmoid':
#             return nnx.sigmoid
#         elif name == 'leaky_relu':
#             return nnx.leaky_relu
#         elif name == 'swish':
#             return nnx.swish
#         elif name == 'elu':
#             return nnx.elu
#         else:
#             raise ValueError("'activation' must be 'relu' or 'tanh'")



class LSTMState(NamedTuple):
    act_state: tuple
    cri_state: tuple

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

    done_prev: jnp.ndarray
    lstm_states_prev: LSTMState

@struct.dataclass
class TrainState:
    graph_def: GraphDef
    state: GraphState

# def construct_net_from_config(config, rng):
#     return RecurrentActorCritic(
#         config["OBSERVATION_SPACE_SIZE"],
#         config["ACTION_SPACE_SIZE"],
#         num_sequences=config["NUM_SEQUENCES"],
#         activation=config["ACTIVATION"],
#         lstm_activation=config["LSTM_ACTIVATION"],
#         net_arch=config.get("NET_ARCH"),
#         act_net_arch=config.get("ACT_NET_ARCH"),
#         cri_net_arch=config.get("CRI_NET_ARCH"),
#         lstm_net_arch=config.get("LSTM_NET_ARCH"),
#         lstm_act_net_arch=config.get("LSTM_ACT_NET_ARCH"),
#         lstm_cri_net_arch=config.get("LSTM_CRI_NET_ARCH"),
#         add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
#         rngs=rng
#     )

def make_train(config, env, env_params):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    config["ACTION_SPACE_SIZE"] = env.action_space(env_params).shape[0]
    config["OBSERVATION_SPACE_SIZE"] = env.observation_space(env_params).shape[0]
    config["NUM_SEQUENCES"] = env.num_obs_sequences

    env = LogWrapper(env)
    # env = ClipAction(env, low=env_params.i_min_action, high=env_params.i_max_action)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac



    _rng = nnx.Rngs(123)
    network = utils.construct_net_from_config(config, _rng)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

    optimizer = nnx.Optimizer(network, tx)

    graph_def, state = nnx.split((network, optimizer))

    train_state = TrainState(graph_def=graph_def, state=state)

    return env, env_params, train_state

@partial(jax.jit, static_argnums=(0, 4, 6))
def test_network(env: MicroGridEnv, env_params, train_state, rng, num_iter, curr_iter=0, print_data=False):

    network, _ = nnx.merge(train_state.graph_def, train_state.state)

    rng, _rng = jax.random.split(rng)

    obsv, env_state = env.reset(_rng, env_params)

    act_state, cri_state = network.get_initial_lstm_state()

    # @scan_tqdm(num_iter, print_rate=num_iter // 100)
    def _env_step(runner_state, unused):
        obsv, env_state, act_state, rng = runner_state

        pi, _, act_state, _ = network(obsv, act_state, cri_state)

        #deterministic action
        action = pi.mode()

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

        runner_state = (obsv, env_state, act_state, rng)
        info['action'] = action
        return runner_state, info

    runner_state = (obsv, env_state, act_state, rng)

    runner_state, info = jax.lax.scan(_env_step, runner_state, jnp.arange(num_iter))

    reward_type = 'weig_reward'

    if print_data:
        jax.debug.print('curr_iter: {i}\n\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}',
                        i=curr_iter, r_tot=jnp.sum(info['r_tot']), r_trad=jnp.sum(info[reward_type]['r_trad']),
                        r_deg=jnp.sum(info[reward_type]['r_deg']), r_clip=jnp.sum(info[reward_type]['r_clipping']))

    return info



@partial(jax.jit, static_argnums=(0, 2, 5, 6, 7, 10), donate_argnums=(3,))
def train(env, env_params, config, train_state, rng, validate=True, freq_val=None, val_env=None, val_params=None, val_rng=None, val_num_iters=None):

    if validate:
        if freq_val is None or val_env is None or val_params is None or val_rng is None or val_num_iters is None:
            raise ValueError("'freq_val', 'val_env', 'val_params', 'val_rng' and 'val_num_iters' must be defined when 'validate' is True")


    rng, _rng = jax.random.split(rng)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])


    ####### INITIALIZE THINGS
    obsv, env_state = env.reset(reset_rng, env_params)

    network, optimizer = nnx.merge(train_state.graph_def, train_state.state)
    lstm_state_act, lstm_state_cri = network.get_initial_lstm_state()

    lstm_state_act, lstm_state_cri = jax.tree.map(lambda x : jnp.tile(x[None, :], (config["NUM_ENVS"],) + (1,) * len(x.shape)), (lstm_state_act, lstm_state_cri))

    # print(lstm_state_act, len(lstm_state_act))
    # print(lstm_state_cri, len(lstm_state_cri))
    #
    # print(jax.tree.map(lambda x: x.shape, lstm_state_act))
    # print(jax.tree.map(lambda x: x.shape, lstm_state_cri))

    pi, _, _, _ = network(obsv, lstm_state_act, lstm_state_cri)
    action = pi.mode()

    _, _, _, done, _ = env.step(
        reset_rng, env_state, action, env_params
    )

    episode_starts = jnp.ones_like(done)

    if validate:
        info = test_network(val_env, val_params, train_state, val_rng, val_num_iters, print_data=False)
        val_info = jax.tree.map(
            lambda x: jnp.empty_like(x, shape=((config['NUM_UPDATES'] - 1) // freq_val + 1,) + x.shape), info)
    else:
        val_info = 0

    # TRAIN LOOP
    @scan_tqdm(config["NUM_UPDATES"], print_rate=1)
    def _update_step(runner_state_plus, curr_iter):
        # COLLECT TRAJECTORIES

        runner_state, val_info = runner_state_plus

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, done_prev, last_lstm_state, rng = runner_state

            network, optimizer = nnx.merge(train_state.graph_def, train_state.state)

            # prev_act_state, prev_cri_state = jnp.where(done_prev,
            #                                            jax.tree.map(lambda x : x[None, :], network.get_initial_lstm_state()),
            #                                            (last_lstm_state.act_state, last_lstm_state.cri_state))

            # prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: done_prev * x[None, :] + y * (1 - done_prev),
            #                                               network.get_initial_lstm_state(),
            #                                               (last_lstm_state.act_state, last_lstm_state.cri_state))

            # print(jax.tree.map(lambda x: x.shape, last_lstm_state.act_state))
            # print(jax.tree.map(lambda x: x.shape, last_lstm_state.cri_state))

            prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: jnp.where(done_prev[(slice(None),) + (None,)*x.ndim], x[None, :], y),
                                                          network.get_initial_lstm_state(),
                                                          (last_lstm_state.act_state, last_lstm_state.cri_state))

            # print(jax.tree.map(lambda x: x.shape, prev_act_state))
            # print(jax.tree.map(lambda x: x.shape, prev_cri_state))

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value, lstm_act_state, lstm_cri_state = network(last_obs, prev_act_state, prev_cri_state)

            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )

            # jax.debug.print('obs: {o}', o=obsv, ordered=True)

            # obsv = obsv / jnp.array([100., 1., 100., 100., 0.0001, 0.0001, 1., 1., 1., 1.])

            # jax.debug.print('obs_norm: {o}', o=obsv, ordered=True)

            info['action'] = action

            # transition = Transition(
            #     done, action, value, reward, log_prob, last_obs, info
            # )

            last_lstm_state = LSTMState(prev_act_state, prev_cri_state)

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
                info=info,

                done_prev=done_prev,
                lstm_states_prev=last_lstm_state)

            # jax.debug.print('{t}', t=transition, ordered=True)

            lstm_states = LSTMState(lstm_act_state, lstm_cri_state)
            runner_state = (train_state, env_state, obsv, done, lstm_states, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # jax.debug.print('{t}', t=jax.tree.map(lambda val: val.shape, traj_batch), ordered=True)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, last_episode_starts, last_lstm_state, rng = runner_state
        network, optimizer = nnx.merge(train_state.graph_def, train_state.state)
        _, last_val, _, _ = network(last_obs, last_lstm_state.act_state, last_lstm_state.cri_state)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )

            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)      #fixme they do the normalization somewhere else, is it the same?

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, sequence_info):
                traj_batch, advantages, targets = sequence_info

                def _loss_fn(network, traj_batch, gae, targets):
                    # RERUN NETWORK

                    init_states = network.get_initial_lstm_state()

                    def forward_pass_lstm(states, data):
                        obs, beginning = data

                        act_state, cri_state = jax.lax.cond(beginning, lambda: init_states, lambda: states)
                        act_state, act_output = network.apply_lstm_act(obs, act_state)
                        cri_state, cri_output = network.apply_lstm_cri(obs, cri_state)
                        return (act_state, cri_state), (act_output, cri_output)
                    #
                    # # print(jax.tree.map(lambda x: x[0].shape, (traj_batch.lstm_states_prev.act_state, traj_batch.lstm_states_prev.act_state)))
                    #

                    normalized_obs = network.normalize_input(traj_batch.obs)
                    _, lstm_output = jax.lax.scan(forward_pass_lstm,
                                                  jax.tree.map(lambda x: x[0], (traj_batch.lstm_states_prev.act_state, traj_batch.lstm_states_prev.cri_state)),
                                                  (normalized_obs, traj_batch.done_prev),
                                                  unroll=16)

                    # prev_act_state, prev_cri_state = jax.tree.map(
                    #     lambda x, y: jnp.where(traj_batch.done_prev[(slice(None),) + (None,) * x.ndim], x[None, :], y),
                    #     init_states,
                    #     (traj_batch.lstm_states_prev.act_state, traj_batch.lstm_states_prev.cri_state))

                    # first_act_state = jax.tree.map(lambda x: x[0], traj_batch.lstm_states_prev.act_state)
                    # first_cri_state = jax.tree.map(lambda x: x[0], traj_batch.lstm_states_prev.cri_state)
                    #
                    # _, outputs = network.apply_lstms_to_sequence(traj_batch.obs, traj_batch.done_prev, first_act_state, first_cri_state, init_states)

                    act_outputs, cri_outputs = lstm_output

                    # pi, values, _, _ = network(traj_batch.obs, prev_act_state, prev_cri_state)



                    # lstm_output_act, lstm_output_cri = lstm_output
                    #
                    pi = network.apply_act_mlp(normalized_obs, act_outputs)
                    values = network.apply_cri_mlp(normalized_obs, cri_outputs)

                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        values - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(values - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)

                    if config["NORMALIZE_ADVANTAGES"]:
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )

                    # jax.debug.print('{t}', t=total_loss, ordered=True)

                    return total_loss, (value_loss, loss_actor, entropy)

                network, optimizer = nnx.merge(train_state.graph_def, train_state.state)

                grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    network, traj_batch, advantages, targets
                )

                optimizer.update(grads)
                train_state = train_state.replace(state=nnx.state((network, optimizer)))
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state

            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"


            # permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)

            #division in sequences
            batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), batch
            )
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((x.shape[0],) + (-1, config['MINIBATCH_SIZE']) + x.shape[2:]), batch
            )
            sequences = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )

            permutation = jax.random.permutation(_rng, config["NUM_MINIBATCHES"])
            shuffled_sequences = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), sequences
            )


            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, shuffled_sequences
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, rng)

        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        train_state = update_state[0]
        metric = traj_batch.info
        rng = update_state[-1]

        if config.get("DEBUG"):

            def callback(info):
                return_values = info["returned_episode_returns"][
                    info["returned_episode"]
                ]
                timesteps = (
                    info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                )
                for t in range(len(timesteps)):
                    print(
                        f"global step={timesteps[t]}, episodic return={return_values[t]}"
                    )

            jax.debug.callback(callback, metric)

        if validate:
            val_info = jax.lax.cond(curr_iter % freq_val == 0,
                                    lambda: jax.tree.map(lambda all, new: all.at[curr_iter // freq_val].set(new),
                                                         val_info,
                                                         test_network(val_env, val_params, train_state, val_rng, val_num_iters, curr_iter=curr_iter, print_data=True)),
                                    lambda: val_info)


        runner_state = (train_state, env_state, last_obs, last_episode_starts, last_lstm_state, rng)

        runner_state_plus = (runner_state, val_info)

        return runner_state_plus, metric

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, episode_starts, LSTMState(lstm_state_act, lstm_state_cri), _rng)

    runner_state_plus = (runner_state, val_info)

    runner_state_plus, metric = jax.lax.scan(
        _update_step, runner_state_plus, jnp.arange(config['NUM_UPDATES'])
    )

    runner_state, val_info = runner_state_plus

    if validate:
        return {'runner_state': runner_state, 'metrics': metric, 'val_info': val_info}
    else:
        return {'runner_state': runner_state, 'metrics': metric}
