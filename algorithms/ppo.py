import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from functools import partial

from jax_tqdm import scan_tqdm

from flax.nnx.nn.initializers import constant, orthogonal, glorot_normal
from flax.nnx import GraphDef, GraphState
import numpy as np
import optax
from typing import Sequence, NamedTuple, Any
import distrax
from .wrappers import (
    LogWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

class ActorCritic(nnx.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, rngs, net_arch: list=None, act_net_arch: list=None, cri_net_arch: list=None):

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

        if activation == 'relu':
            activation = nnx.relu
        else:
            activation = nnx.tanh

        act_net_arch = [in_features] + act_net_arch + [out_features]

        self.act_layers = []
        for i in range(len(act_net_arch) - 2):
            self.act_layers.append(nnx.Linear(act_net_arch[i], act_net_arch[i+1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.), rngs=rngs))
            self.act_layers.append(activation)
        self.act_layers.append(nnx.Linear(act_net_arch[-2], act_net_arch[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.), rngs=rngs))


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

def make_train(config, env, env_params):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
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
    network = ActorCritic(
        env.observation_space(env_params).shape[0],
        env.action_space(env_params).shape[0],
        activation=config["ACTIVATION"],
        net_arch=config.get("NET_ARCH"),
        act_net_arch=config.get("ACT_NET_ARCH"),
        cri_net_arch=config.get("CRI_NET_ARCH"),
        rngs=_rng
    )

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

@partial(jax.jit, static_argnums=(0, 2), donate_argnums=(3,))
def train(env, env_params, config, train_state, rng):
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)

    # TRAIN LOOP
    @scan_tqdm(config["NUM_UPDATES"], print_rate=5)
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
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
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

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

        runner_state = (train_state, env_state, last_obs, rng)

        return runner_state, metric

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, _rng)

    runner_state, metric = jax.lax.scan(
        _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
    )

    return {"runner_state": runner_state, "metrics": metric}