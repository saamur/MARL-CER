import jax
import jax.numpy as jnp
from jax.experimental import io_callback
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
from ernestogym.envs.single_agent.env import MicroGridEnv

from .wrappers import (
    LogWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

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

def make_train(config, env, env_params, seed=123):
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


    _rng = nnx.Rngs(seed)
    network = utils.construct_net_from_config(config, _rng)
    network.eval()

    def schedule_builder(lr_init, lr_end):

        tot_steps = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"] * config["NUM_UPDATES"]

        if config["LR_SCHEDULE"] == 'linear':
            return optax.schedules.linear_schedule(lr_init, lr_end, tot_steps)
        elif config["LR_SCHEDULE"] == 'cosine':
            return optax.schedules.cosine_decay_schedule(lr_init, tot_steps, lr_end / lr_init)
        else:
            return lr_init

    schedule = schedule_builder(config['LR'], config['LR_MIN'])

    if config['USE_WEIGHT_DECAY']:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=schedule, eps=1e-5),
        )

    optimizer = nnx.Optimizer(network, tx)

    graph_def, state = nnx.split((network, optimizer))

    train_state = TrainState(graph_def=graph_def, state=state)

    return env, env_params, train_state

@partial(jax.jit, static_argnums=(0, 4, 6))
def test_network(env: MicroGridEnv, env_params, train_state, rng, num_iter, curr_iter=0, print_data=False):

    network, _ = nnx.merge(train_state.graph_def, train_state.state)

    rng, _rng = jax.random.split(rng)

    env_params = env.eval(env_params)

    obsv, env_state = env.reset(_rng, env_params)

    env_params = env_params.replace(test_profile=env_params.test_profile + 1)

    # @scan_tqdm(num_iter, print_rate=num_iter // 100)
    def _env_step(runner_state, unused):
        obsv, env_state, env_params, rng = runner_state

        pi, _ = network(obsv)

        #deterministic action
        action = pi.mode()

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

        rng, _rng = jax.random.split(rng)
        env_params = jax.lax.cond(done,
                                  lambda: env_params.replace(test_profile=env_params.test_profile + 1),
                                  lambda: env_params)

        # jax.lax.cond(done, lambda: jax.debug.print('i {x}, {y}', x=unused, y=env_params.test_profile), lambda : None)

        # jax.lax.cond(done, lambda: jax.debug.print('i {x}, {dem}, {gen}, {spr}, {bpr}, {rew}, {pr}, {nr}, {wr}\n{soh}\n',
        #                                            x=unused, dem=info['demand'], gen=info['generation'],
        #                                            spr=info['sell_price'], bpr=info['buy_price'], rew=info['r_tot'],
        #                                            pr=info['pure_reward'], nr=info['norm_reward'], wr=info['weig_reward'],
        #              soh=info['soh'], ordered=True),
        #              lambda: None)

        runner_state = (obsv, env_state, env_params, rng)
        info['action'] = action
        return runner_state, info

    runner_state = (obsv, env_state, env_params, rng)

    runner_state, info = jax.lax.scan(_env_step, runner_state, jnp.arange(num_iter))

    reward_type = 'weig_reward'

    if print_data:
        jax.debug.print('curr_iter: {i}\n\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}'
                        '\n\tmean soc: {mean_soc}\n\tstd actions: {std_act}\n',
                        i=curr_iter, r_tot=jnp.sum(info['r_tot']), r_trad=jnp.sum(info[reward_type]['r_trad']), r_deg=jnp.sum(info[reward_type]['r_deg']), r_clip=jnp.sum(info[reward_type]['r_clipping']),
                        mean_soc=jnp.mean(info['soc']), std_act=jnp.std(info['action']))

    return info

# @partial(jax.jit, static_argnums=(0, 2, 5, 6, 7, 10), donate_argnums=(3,))
def train_wrapper(env, env_params, config, train_state, rng, validate=True, freq_val=None, val_env=None, val_params=None, val_rng=None, val_num_iters=None):

    infos = {}
    val_infos = {}

    def end_update_step(info, i):
        if len(infos) == 0:
            # infos = jax.tree.map(lambda x: np.empty_like(x, shape=(config["NUM_UPDATES"],)+x.shape), info)
            infos.update(jax.tree.map(lambda x: np.empty_like(x, shape=(config['NUM_UPDATES'],)+x.shape), info))

        info = jax.device_put(info, device=jax.devices('cpu')[0])

        def update(logs, new):
            logs[i] = new

        jax.tree.map(update, infos, info)

    def update_val_info(val_info, i):
        if len(val_infos) == 0:
            # infos = jax.tree.map(lambda x: np.empty_like(x, shape=(config["NUM_UPDATES"],)+x.shape), info)
            val_infos.update(jax.tree.map(lambda x: np.empty_like(x, shape=((config['NUM_UPDATES'] - 1) // freq_val + 1,) + x.shape), val_info))

        def update(logs, new):
            logs[i] = new

        val_info = jax.device_put(val_info, device=jax.devices('cpu')[0])
        jax.tree.map(update, val_infos, val_info)


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
            # val_info = jax.device_get(val_info)
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
                def _get_advantages(gae_and_next_value, transition_data):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition_data
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                rewards = traj_batch.reward

                if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS']:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    (traj_batch.done, traj_batch.value, rewards),
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

                        if config['NORMALIZE_TARGETS']:
                            targets = (targets - targets.mean()) / (targets.std() + 1e-8)

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

                        # jax.debug.print('ratio mean {x}, max {y}, min {z}, std {w}', x=ratio.mean(), y=ratio.max(), z=ratio.min(), w=ratio.std(), ordered=True)

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

                        # jax.debug.print('val {x}, act {y}, ent {z}', x=value_loss, y=loss_actor, z=entropy, ordered=True)
                        return total_loss, (value_loss, loss_actor, entropy)

                    network, optimizer = nnx.merge(train_state.graph_def, train_state.state)

                    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        network, traj_batch, advantages, targets
                    )

                    # jax.debug.print('grad norm {x}', x=optax.global_norm(grads), ordered=True)

                    optimizer.update(grads)
                    # jax.debug.print('log std {x}', x=network.log_std.value, ordered=True)

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

            network, optimizer = nnx.merge(train_state.graph_def, train_state.state)
            network.train()
            graph_def, state = nnx.split((network, optimizer))
            train_state = TrainState(graph_def=graph_def, state=state)

            update_state = (train_state, traj_batch, advantages, targets, rng)

            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]

            network, optimizer = nnx.merge(train_state.graph_def, train_state.state)
            network.eval()
            graph_def, state = nnx.split((network, optimizer))
            train_state = TrainState(graph_def=graph_def, state=state)

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
                _ = jax.lax.cond(curr_iter % freq_val == 0,
                                 lambda : io_callback(update_val_info, None, test_network(val_env, val_params, train_state, val_rng, val_num_iters, curr_iter=curr_iter, print_data=True), curr_iter // freq_val, ordered=True),
                                 lambda : None)

            runner_state = (train_state, env_state, last_obs, rng)

            runner_state_plus = (runner_state, val_info)

            # metric = jax.device_put(metric, device=jax.devices('cpu')[0])

            io_callback(end_update_step, None, metric, curr_iter, ordered=True)
            # metric = jax.device_get(metric)

            return runner_state_plus, None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)

        runner_state_plus = (runner_state, val_info)

        runner_state_plus, _ = jax.lax.scan(
            _update_step, runner_state_plus, jnp.arange(config["NUM_UPDATES"])
        )

        runner_state, val_info = runner_state_plus

        return runner_state

        # if validate:
        #     return jax.device_put({'runner_state': runner_state, 'metrics': metric, 'val_info': val_info}, device=jax.devices('cpu')[0])
        # else:
        #     return jax.device_get({'runner_state': runner_state, 'metrics': metric})

    runner_state = train(env, env_params, config, train_state, rng, validate, freq_val, val_env, val_params, val_rng, val_num_iters)

    # metric = jax.tree.map(lambda *vals: jnp.stack(vals, axis=0), *infos)

    if validate:
        # val_info = jax.tree.map(lambda *vals: jnp.stack(vals, axis=0), *val_infos)
        return jax.device_put({'runner_state': runner_state, 'metrics': infos, 'val_info': val_infos}, device=jax.devices('cpu')[0])
    else:
        return jax.device_get({'runner_state': runner_state, 'metrics': infos})