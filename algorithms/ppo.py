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

import algorithms.utils as utils
from ernestogym.envs_jax.single_agent.env import MicroGridEnv

from .wrappers import (
    LogWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

class ActorCritic(nnx.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, rngs, net_arch: list=None, act_net_arch: list=None, cri_net_arch: list=None, add_logistic_to_actor: bool = False):

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

        activation = self.activation_from_name(activation)

        act_net_arch = [in_features] + act_net_arch + [out_features]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))
        if add_logistic_to_actor:
            self.act_layers.append(nnx.sigmoid)

        self.log_std = nnx.Param(jnp.zeros(out_features))

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

        actor_mean = x
        for layer in self.act_layers:
            actor_mean = layer(actor_mean)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(jnp.asarray(self.log_std)))

        critic = x
        for layer in self.cri_layers:
            critic = layer(critic)

        return pi, jnp.squeeze(critic, axis=-1)

    @classmethod
    def activation_from_name(cls, name: str):
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

@struct.dataclass
class TrainState:
    graph_def: GraphDef
    state: GraphState

# def construct_net_from_config(config, rng):
#     return ActorCritic(
#     config["OBSERVATION_SPACE_SIZE"],
#     config["ACTION_SPACE_SIZE"],
#     activation=config["ACTIVATION"],
#     net_arch=config.get("NET_ARCH"),
#     act_net_arch=config.get("ACT_NET_ARCH"),
#     cri_net_arch=config.get("CRI_NET_ARCH"),
#     add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
#     rngs=rng
# )

def make_train(config, env, env_params):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    config["ACTION_SPACE_SIZE"] = env.action_space(env_params).shape[0]
    config["OBSERVATION_SPACE_SIZE"] = env.observation_space(env_params).shape[0]

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

    # @scan_tqdm(num_iter, print_rate=num_iter // 100)
    def _env_step(runner_state, unused):
        obsv, env_state, rng = runner_state

        pi, _ = network(obsv)

        #deterministic action
        action = pi.mode()

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

        runner_state = (obsv, env_state, rng)
        info['action'] = action
        return runner_state, info

    runner_state = (obsv, env_state, rng)

    runner_state, info = jax.lax.scan(_env_step, runner_state, jnp.arange(num_iter))

    reward_type = 'weig_reward'

    if print_data:
        jax.debug.print('curr_iter: {i}\n\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}',
                        i=curr_iter, r_tot=jnp.sum(info['r_tot']), r_trad=jnp.sum(info[reward_type]['r_trad']), r_deg=jnp.sum(info[reward_type]['r_deg']), r_clip=jnp.sum(info[reward_type]['r_clipping']))

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
    obsv, env_state = env.reset(reset_rng, env_params)

    if validate:
        info = test_network(val_env, val_params, train_state, val_rng, val_num_iters, print_data=False)
        val_info = jax.tree.map(lambda x: jnp.empty_like(x, shape=((config['NUM_UPDATES']-1)//freq_val+1,)+x.shape), info)
    else:
        val_info = 0

    # TRAIN LOOP
    @scan_tqdm(config["NUM_UPDATES"], print_rate=5)
    def _update_step(runner_state_plus, curr_iter):
        # COLLECT TRAJECTORIES

        runner_state, val_info = runner_state_plus

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            network, optimizer = nnx.merge(train_state.graph_def, train_state.state)

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network(last_obs)

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

            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )

            # jax.debug.print('{t}', t=transition, ordered=True)
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # jax.debug.print('{t}', t=jax.tree.map(lambda val: val.shape, traj_batch), ordered=True)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng = runner_state
        network, optimizer = nnx.merge(train_state.graph_def, train_state.state)
        _, last_val = network(last_obs)

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

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(network, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = network(traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = jax.lax.cond(config["NORMALIZE_ADVANTAGES"],
                                       lambda : (gae - gae.mean()) / (gae.std() + 1e-8),
                                       lambda : gae)
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
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)

            # jax.debug.print('bef {z}', z=jax.tree.map(lambda l: l.shape, traj_batch), ordered=True)

            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            # jax.debug.print('aft {z}', z=jax.tree.map(lambda l: l.shape, batch[0]), ordered=True)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # jax.debug.print('aft2 {z}', z=jax.tree.map(lambda l: l.shape, shuffled_batch[0]), ordered=True)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            # jax.debug.print('aft3 {z}', z=jax.tree.map(lambda l: l.shape, minibatches[0]), ordered=True)
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
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

        runner_state = (train_state, env_state, last_obs, rng)

        runner_state_plus = (runner_state, val_info)

        return runner_state_plus, metric

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, _rng)

    runner_state_plus = (runner_state, val_info)

    runner_state_plus, metric = jax.lax.scan(
        _update_step, runner_state_plus, jnp.arange(config["NUM_UPDATES"])
    )

    runner_state, val_info = runner_state_plus

    if validate:
        return {'runner_state': runner_state, 'metrics': metric, 'val_info': val_info}
    else:
        return {'runner_state': runner_state, 'metrics': metric}




@partial(jax.jit, static_argnums=(0, 2), donate_argnums=(3,))
def train_for(env, env_params, config, train_state, rng):
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])



    #BUILD TRAJECTORY SHAPE
    obsv, env_state = env.reset(reset_rng, env_params)

    network, optimizer = nnx.merge(train_state.graph_def, train_state.state)

    rng, _rng = jax.random.split(rng)
    pi, value = network(obsv)

    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    obsv, env_state, reward, done, info = env.step(
        reset_rng, env_state, action, env_params
    )

    info['action'] = action

    transition_lalala = Transition(done, action, value, reward, log_prob, obsv, info)

    # transition_shape_and_dtype = jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), transition_lalala)
    # # transition_dtype = jax.tree.map(lambda leaf: leaf.dtype, transition_lalala)
    #
    # jax.debug.print('{a}', a=tuple(transition_shape_and_dtype), ordered=True)
    # jax.debug.print('{a}', a=transition_dtype, ordered=True)
    ###############################################

    obsv, env_state = env.reset(reset_rng, env_params)

    # TRAIN LOOP
    @loop_tqdm(config["NUM_UPDATES"], print_rate=5)
    def _update_step(i, val):

        runner_state, metric = val

        # COLLECT TRAJECTORIES
        def _env_step(j, val):
            runner_state, traj_batch = val
            train_state, env_state, last_obs, rng = runner_state

            network, optimizer = nnx.merge(train_state.graph_def, train_state.state)

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network(last_obs)

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

            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )

            traj_batch = jax.tree.map(lambda batch, tr: batch.at[j].set(tr), traj_batch, transition)

            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, traj_batch

        n_steps = config["NUM_STEPS"]
        # jax.debug.print('{t}', t=n_steps, ordered=True)

        # traj_batch = jax.tree.map(lambda shape, dtype: jnp.empty(shape=(n_steps,)+shape, dtype=dtype), transition_shape, transition_dtype)
        traj_batch = jax.tree.map(lambda l: jnp.empty(shape=(n_steps,) + l.shape, dtype=l.dtype), transition_lalala)
        # traj_batch = jax.tree.map(lambda val: jnp.empty(shape=(n_steps,) + val[0], dtype=val[1]), transition_shape_and_dtype)

        runner_state, traj_batch = jax.lax.fori_loop(0, n_steps, _env_step, (runner_state, traj_batch))

        # jax.debug.print('{t}', t=jax.tree.map(lambda val: val.shape, traj_batch), ordered=True)

        # runner_state, traj_batch = jax.lax.scan(
        #     _env_step, runner_state, None, config["NUM_STEPS"]
        # )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng = runner_state
        network, optimizer = nnx.merge(train_state.graph_def, train_state.state)
        _, last_val = network(last_obs)

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

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(j, val):

            update_state, loss_info = val

            def _update_minbatch(k, val):
                train_state, total_loss_batch, minibatches = val
                batch_info = jax.tree.map(lambda leaf: leaf[k], minibatches)

                traj_batch, advantages, targets = batch_info

                def _loss_fn(network, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = network(traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = jax.lax.cond(config["NORMALIZE_ADVANTAGES"],
                                       lambda : (gae - gae.mean()) / (gae.std() + 1e-8),
                                       lambda : gae)
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
                    return total_loss, (value_loss, loss_actor, entropy)

                network, optimizer = nnx.merge(train_state.graph_def, train_state.state)

                grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    network, traj_batch, advantages, targets
                )

                optimizer.update(grads)
                train_state = train_state.replace(state=nnx.state((network, optimizer)))

                total_loss_batch = jax.tree.map(lambda batch, loss: batch.at[k].set(loss), total_loss_batch, total_loss)

                return train_state, total_loss_batch, minibatches

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            # jax.debug.print('minibatches: {m}', m=jax.tree.map(lambda val: tuple(val.shape), minibatches))

            n_minibatch = config["NUM_MINIBATCHES"]

            total_loss_batch = (jnp.empty((n_minibatch,)), (jnp.empty((n_minibatch,)), jnp.empty((n_minibatch,)), jnp.empty((n_minibatch,))))

            train_state, total_loss, minibatches = jax.lax.fori_loop(0, n_minibatch, _update_minbatch, (train_state, total_loss_batch, minibatches))

            # train_state, total_loss = jax.lax.scan(
            #     _update_minbatch, train_state, minibatches
            # )

            # jax.debug.print('total_loss: {m}', m=jax.tree.map(lambda val: tuple(val.shape), total_loss))
            # jax.debug.print('total_loss: {m}', m=type(total_loss))
            update_state = (train_state, traj_batch, advantages, targets, rng)

            loss_info = jax.tree.map(lambda batch, loss: batch.at[j].set(loss), loss_info, total_loss)

            return update_state, loss_info

        n_minibatch = config["NUM_MINIBATCHES"]
        n_epochs = config["UPDATE_EPOCHS"]

        loss_info = (jnp.empty((n_epochs, n_minibatch)), (jnp.empty((n_epochs, n_minibatch)), jnp.empty((n_epochs, n_minibatch)), jnp.empty((n_epochs, n_minibatch))))

        update_state = (train_state, traj_batch, advantages, targets, rng)

        update_state, loss_info = jax.lax.fori_loop(0, n_epochs, _update_epoch, (update_state, loss_info))

        # update_state, loss_info = jax.lax.scan(
        #     _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        # )

        train_state = update_state[0]
        # metric = traj_batch.info
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

        runner_state = (train_state, env_state, last_obs, rng)

        metric = jax.tree.map(lambda batch, loss: batch.at[i].set(loss), metric, traj_batch.info)

        return runner_state, metric

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, _rng)

    n_steps = config["NUM_STEPS"]
    n_updates = config["NUM_UPDATES"]

    metric = jax.tree.map(lambda l: jnp.empty(shape=(n_updates, n_steps) + l.shape, dtype=l.dtype), transition_lalala.info)

    runner_state, metric = jax.lax.fori_loop(0, n_updates, _update_step, (runner_state, metric))
    # runner_state, metric = jax.lax.scan(
    #     _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
    # )

    return {"runner_state": runner_state, "metrics": metric}



@partial(nnx.jit, static_argnums=(0, 2), donate_argnums=(3,))
def train_for_flax(env, env_params, config, network, optimizer, rng):
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])


    #BUILD TRAJECTORY SHAPE
    obsv, env_state = env.reset(reset_rng, env_params)

    rng, _rng = jax.random.split(rng)
    pi, value = network(obsv)

    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    obsv, env_state, reward, done, info = env.step(
        reset_rng, env_state, action, env_params
    )

    info['action'] = action

    transition_lalala = Transition(done, action, value, reward, log_prob, obsv, info)

    # transition_shape_and_dtype = jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), transition_lalala)
    # # transition_dtype = jax.tree.map(lambda leaf: leaf.dtype, transition_lalala)
    #
    # jax.debug.print('{a}', a=tuple(transition_shape_and_dtype), ordered=True)
    # jax.debug.print('{a}', a=transition_dtype, ordered=True)
    ###############################################

    obsv, env_state = env.reset(reset_rng, env_params)

    # TRAIN LOOP
    # @loop_tqdm(config["NUM_UPDATES"], print_rate=5)
    def _update_step(i, val):

        runner_state, metric = val

        # COLLECT TRAJECTORIES
        def _env_step(j, val):
            runner_state, traj_batch = val
            network, env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network(last_obs)

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

            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )

            traj_batch = jax.tree.map(lambda batch, tr: batch.at[j].set(tr), traj_batch, transition)

            runner_state = (network, env_state, obsv, rng)
            return runner_state, traj_batch

        n_steps = config["NUM_STEPS"]
        # jax.debug.print('{t}', t=n_steps, ordered=True)

        # traj_batch = jax.tree.map(lambda shape, dtype: jnp.empty(shape=(n_steps,)+shape, dtype=dtype), transition_shape, transition_dtype)
        traj_batch = jax.tree.map(lambda l: jnp.empty(shape=(n_steps,) + l.shape, dtype=l.dtype), transition_lalala)
        # traj_batch = jax.tree.map(lambda val: jnp.empty(shape=(n_steps,) + val[0], dtype=val[1]), transition_shape_and_dtype)

        runner_state, traj_batch = nnx.fori_loop(0, n_steps, _env_step, (runner_state, traj_batch))

        # jax.debug.print('{t}', t=jax.tree.map(lambda val: val.shape, traj_batch), ordered=True)

        # runner_state, traj_batch = jax.lax.scan(
        #     _env_step, runner_state, None, config["NUM_STEPS"]
        # )

        # CALCULATE ADVANTAGE
        network, env_state, last_obs, rng = runner_state
        _, last_val = network(last_obs)

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

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(j, val):

            update_state, loss_info = val

            def _update_minbatch(k, val):
                network, total_loss_batch, minibatches = val
                batch_info = jax.tree.map(lambda leaf: leaf[k], minibatches)

                traj_batch, advantages, targets = batch_info

                def _loss_fn(network, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = network(traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = jax.lax.cond(config["NORMALIZE_ADVANTAGES"],
                                       lambda : (gae - gae.mean()) / (gae.std() + 1e-8),
                                       lambda : gae)
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
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    network, traj_batch, advantages, targets
                )

                optimizer.update(grads)
                total_loss_batch = jax.tree.map(lambda batch, loss: batch.at[k].set(loss), total_loss_batch, total_loss)

                return network, total_loss_batch, minibatches

            network, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            # jax.debug.print('minibatches: {m}', m=jax.tree.map(lambda val: tuple(val.shape), minibatches))

            n_minibatch = config["NUM_MINIBATCHES"]

            total_loss_batch = (jnp.empty((n_minibatch,)), (jnp.empty((n_minibatch,)), jnp.empty((n_minibatch,)), jnp.empty((n_minibatch,))))

            network, total_loss, minibatches = nnx.fori_loop(0, n_minibatch, _update_minbatch, (network, total_loss_batch, minibatches))

            # jax.debug.print('total_loss: {m}', m=jax.tree.map(lambda val: tuple(val.shape), total_loss))
            # jax.debug.print('total_loss: {m}', m=type(total_loss))
            update_state = (network, traj_batch, advantages, targets, rng)

            loss_info = jax.tree.map(lambda batch, loss: batch.at[j].set(loss), loss_info, total_loss)

            return update_state, loss_info

        n_minibatch = config["NUM_MINIBATCHES"]
        n_epochs = config["UPDATE_EPOCHS"]

        loss_info = (jnp.empty((n_epochs, n_minibatch)), (jnp.empty((n_epochs, n_minibatch)), jnp.empty((n_epochs, n_minibatch)), jnp.empty((n_epochs, n_minibatch))))

        update_state = (network, traj_batch, advantages, targets, rng)

        update_state, loss_info = nnx.fori_loop(0, n_epochs, _update_epoch, (update_state, loss_info))

        # metric = traj_batch.info
        network = update_state[0]
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

        runner_state = (network, env_state, last_obs, rng)

        metric = jax.tree.map(lambda batch, loss: batch.at[i].set(loss), metric, traj_batch.info)

        return runner_state, metric

    rng, _rng = jax.random.split(rng)
    runner_state = (network, env_state, obsv, _rng)

    n_steps = config["NUM_STEPS"]
    n_updates = config["NUM_UPDATES"]

    metric = jax.tree.map(lambda l: jnp.empty(shape=(n_updates, n_steps) + l.shape, dtype=l.dtype), transition_lalala.info)

    runner_state, metric = nnx.fori_loop(0, n_updates, _update_step, (runner_state, metric))

    return {"runner_state": runner_state, "metrics": metric}