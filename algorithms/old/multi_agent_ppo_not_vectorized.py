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
from ernestogym.envs.multi_agent.env import RECEnv
from algorithms.ppo import ActorCritic

from .wrappers import (
    LogWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

class StackedActorCritic(ActorCritic):
    @nnx.vmap(in_axes=(0, None, None, None, 0, None, None, None, None))
    def __init__(self, in_features: int, out_features: int, activation: str, rngs, net_arch: list, act_net_arch: list, cri_net_arch: list, add_logistic_to_actor: bool):
        super().__init__(in_features, out_features, activation, rngs, net_arch=net_arch, act_net_arch=act_net_arch, cri_net_arch=cri_net_arch, add_logistic_to_actor=add_logistic_to_actor)

    @nnx.vmap
    def __call__(self, x):
        #distrax does not support vmap well
        #specifically distrax.MultivariateNormalDiag is cited in the README to give some problems
        #so before leaving vmap I return the arrays needed to reconstruct the distribution outside
        pi, val = super().__call__(x)

        return pi.loc, pi.scale_diag, val

class WrappedStackedActorCritic(nnx.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, rngs, net_arch: list, act_net_arch: list,
                 cri_net_arch: list, add_logistic_to_actor: bool):
        self.vmappd_module = StackedActorCritic(in_features, out_features, activation, rngs, net_arch=net_arch, act_net_arch=act_net_arch, cri_net_arch=cri_net_arch, add_logistic_to_actor=add_logistic_to_actor)

    def __call__(self, x):
        pi_loc, pi_scale, val = self.vmappd_module(x)
        pi = distrax.MultivariateNormalDiag(pi_loc, pi_scale)
        return pi, val

class RECActorCritic(nnx.Module):
    def __init__(self, in_features: int, num_battery_agents: int, activation: str, rngs, net_arch: list=None, act_net_arch: list=None, cri_net_arch: list=None, passive_houses: bool=False):

        if act_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'act_net_arch' is None")
            act_net_arch = net_arch
        if cri_net_arch is None:
            if net_arch is None:
                raise ValueError("'net_arch' must be specified if 'cri_net_arch' is None")
            cri_net_arch = net_arch

        self.passive_houses = passive_houses

        act_net_arch = list(act_net_arch)
        cri_net_arch = list(cri_net_arch)

        activation = self.activation_from_name(activation)

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

        # jax.debug.print('data {x}', x=data)

        logit = data
        for layer in self.act_layers:
            # jax.debug.print('aa {x}', x=logit)
            logit = layer(logit)

        alpha = nnx.softplus(logit).squeeze(axis=-1) + 1e-6
        # jax.debug.print('{x}', x=alpha)
        # alpha = jnp.clip(alpha, max=1e+4)
        # jax.debug.print('{x}', x=alpha)

        pi = distrax.Dirichlet(alpha)

        critic = data
        for layer in self.cri_layers:
            critic = layer(critic)

        return pi, jnp.squeeze(critic, axis=-1)


    def prepare_data(self, obs):
        demands_base = obs['demands_base_battery_houses']
        demands_batteries = obs['demands_battery_battery_houses']
        generations = obs['generations_battery_houses']
        num_battery_agents = demands_batteries.shape[-1]
        if self.passive_houses:
            demands_base = jnp.concat((demands_base, obs['demand_passive_houses']), axis=-1)
            demands_batteries = jnp.concat((demands_batteries, jnp.zeros_like(obs['demand_passive_houses'])), axis=-1)
            generations = jnp.concat((generations, obs['generations_passive_houses']), axis=-1)

        demand_base_rec = jnp.sum(demands_base, axis=-1)
        demand_batteries_rec = jnp.sum(demands_batteries, axis=-1)
        generation_rec = jnp.sum(generations, axis=-1)

        network_exchange = generations - demands_base - demands_batteries

        network_rec_plus = jnp.sum(jnp.maximum(network_exchange, 0.), axis=-1)
        network_rec_minus = -jnp.sum(jnp.minimum(network_exchange, 0.), axis=-1)

        global_data = jnp.stack(
            [demand_base_rec, demand_batteries_rec, generation_rec, network_rec_plus, network_rec_minus,
             obs['sin_seconds_of_day'], obs['cos_seconds_of_day'], obs['sin_day_of_year'], obs['cos_day_of_year']],
            axis=-1)

        global_data = jnp.expand_dims(global_data, -2)
        global_data = jnp.repeat(global_data, num_battery_agents, axis=-2)

        local_data = jnp.stack([obs['demands_base_battery_houses'], obs['demands_battery_battery_houses'],
                                obs['generations_battery_houses']], axis=-1)

        data = jnp.concatenate((global_data, local_data), axis=-1)

        return data






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
    done_batteries: jnp.ndarray
    done_rec: jnp.ndarray

    actions_batteries: jnp.ndarray      #
    actions_rec: jnp.ndarray

    values_batteries: jnp.ndarray       #
    value_rec: jnp.ndarray

    rewards_batteries: jnp.ndarray
    reward_rec: jnp.ndarray

    log_prob_batteries: jnp.ndarray     #
    log_prob_rec: jnp.ndarray

    obs_batteries: jnp.ndarray      #
    obs_rec: jnp.ndarray

    info: jnp.ndarray

@struct.dataclass
class TrainState:
    graph_def: GraphDef
    state: GraphState

def make_train(config, env):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    config["BATTERY_ACTION_SPACE_SIZE"] = env.action_space(env.battery_agents[0]).shape[0]
    config["BATTERY_OBSERVATION_SPACE_SIZE"] = env.observation_space(env.battery_agents[0]).shape[0]

    config["REC_ACTION_SPACE_SIZE"] = env.action_space(env.rec_agent).shape[0]
    config["REC_INPUT_NETWORK_SIZE"] = 12        #fixme bruttissimo hardcoded
    config['NUM_BATTERY_AGENTS'] = env.num_battery_agents
    config['PASSIVE_HOUSES'] = (env.num_passive_houses>0)

    # env = LogWrapper(env)
    # env = ClipAction(env, low=env_params.i_min_action, high=env_params.i_max_action)
    # if config["NORMALIZE_ENV"]:
    #     env = NormalizeVecObservation(env)
    #     env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def linear_schedule_bat(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * config['NUM_BATTERY_AGENTS'] * frac



    _rng = nnx.Rngs(123)
    network_batteries = utils.construct_battery_net_from_config_multi_agent_only_actor_critic(config, _rng)
    _rng = nnx.Rngs(222)
    network_rec = utils.construct_rec_net_from_config_multi_agent_only_actor_critic(config, _rng)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        tx_bat = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule_bat, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        tx_bat=tx

    optimizer_batteries = nnx.Optimizer(network_batteries, tx_bat)
    optimizer_rec = nnx.Optimizer(network_rec, tx)

    graph_def, state = nnx.split((network_batteries, optimizer_batteries, network_rec, optimizer_rec))

    train_state = TrainState(graph_def=graph_def, state=state)

    return env, train_state

# @partial(jax.jit, static_argnums=(0, 4, 6))
def test_network(env: RECEnv, env_params, train_state, rng, num_iter, curr_iter=0, print_data=False):

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

@partial(jax.jit, static_argnums=(0, 1, 4, 5, 6, 9), donate_argnums=(3,))
def train(env: RECEnv, config, train_state, rng, validate=True, freq_val=None, val_env=None, val_params=None, val_rng=None, val_num_iters=None):

    if validate:
        if freq_val is None or val_env is None or val_params is None or val_rng is None or val_num_iters is None:
            raise ValueError("'freq_val', 'val_env', 'val_params', 'val_rng' and 'val_num_iters' must be defined when 'validate' is True")

    rng, _rng = jax.random.split(rng)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    # reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(_rng)

    if validate:
        info = test_network(val_env, val_params, train_state, val_rng, val_num_iters, print_data=False)
        val_info = jax.tree.map(lambda x: jnp.empty_like(x, shape=((config['NUM_UPDATES']-1)//freq_val+1,)+x.shape), info)
    else:
        val_info = 0

    # TRAIN LOOP
    @scan_tqdm(config["NUM_UPDATES"], print_rate=min(config["NUM_UPDATES"], 5))
    def _update_step(runner_state_plus, curr_iter):
        # COLLECT TRAJECTORIES

        runner_state, val_info = runner_state_plus

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs_batteries, rng = runner_state

            networks_batteries, optimizer_batteries, network_rec, optimizer_rec = nnx.merge(train_state.graph_def, train_state.state)

            # print(type(networks_batteries), type(optimizer_batteries), type(network_rec), type(optimizer_rec))

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            # _rng = jax.random.split(_rng, num=env.num_battery_agents)

            # print(f'aaa {config['BATTERY_OBSERVATION_SPACE_SIZE']}')
            # print(last_obs_batteries.shape)

            pi, value_batteries = networks_batteries(last_obs_batteries)

            actions_batteries = pi.sample(seed=_rng)
            log_prob_batteries = pi.log_prob(actions_batteries)

            # print(f'gddbfddbgffxgn {actions_batteries.shape}')

            # actions_batteries = jnp.zeros((env.num_battery_agents,))
            # log_prob_batteries = jnp.ones_like(actions_batteries)

            actions_first = {env.battery_agents[i]: actions_batteries[i] for i in range(env.num_battery_agents)}#zip(env.battery_agents, actions_batteries)}
            actions_first[env.rec_agent] = jnp.zeros(env.num_battery_agents)

            # print(actions_first)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward_first, done_first, info_first = env.step(
                _rng, env_state, actions_first
            )

            info_first['actions'] = actions_first


            rec_obsv = obsv[env.rec_agent]
            rng, _rng = jax.random.split(rng)

            # print(f'rec_obs {jax.tree.map(lambda x: x.shape, rec_obsv)}')

            pi, value_rec = network_rec(rec_obsv)

            actions_rec = pi.sample(seed=_rng)
            # actions_rec = jnp.ones(env.num_battery_agents) / env.num_battery_agents

            # jax.debug.print('curr iter {i} actions: {a}', i=curr_iter, a=actions_rec)

            log_probs_rec = pi.log_prob(actions_rec)

            actions_second = {agent: jnp.array([0.]) for agent in env.battery_agents}
            actions_second[env.rec_agent] = actions_rec

            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward_second, done_second, info_second = env.step(
                _rng, env_state, actions_second
            )

            info_second['actions'] = actions_second

            done = jax.tree.map(jnp.logical_or, done_first, done_second)
            done_batteries = jnp.array([done[a] for a in env.battery_agents])
            done_rec = done[env.rec_agent]

            rewards_tot = jax.tree_map(lambda x, y: x + y, reward_first, reward_second)
            rewards_batteries = jnp.array([rewards_tot[a] for a in env.battery_agents])
            reward_rec = rewards_tot[env.rec_agent]

            info = jax.tree.map(lambda  x, y: x + y, info_first, info_second)

            obs_batteries = jnp.vstack([obsv[a] for a in env.battery_agents])


            transition = Transition(
                done_batteries, done_rec,
                actions_batteries, actions_rec,
                value_batteries, value_rec,
                rewards_batteries, reward_rec,
                log_prob_batteries, log_probs_rec,
                last_obs_batteries, rec_obsv,
                info
            )

            # jax.debug.print('{t}', t=transition, ordered=True)
            runner_state = (train_state, env_state, obs_batteries, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # jax.debug.print('{t}', t=jax.tree.map(lambda val: val.shape, traj_batch), ordered=True)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs_batteries, rng = runner_state
        networks_batteries, optimizer_batteries, network_rec, optimizer_rec = nnx.merge(train_state.graph_def, train_state.state)
        _, last_val_batteries = networks_batteries(last_obs_batteries)
        last_val_rec = traj_batch.value_rec[-1]

        def _calculate_gae(traj_batch, last_val_batteries, last_val_rec):
            def _get_advantages(gae_and_next_value, transition):
                gae_batteries, next_value_batteries, gae_rec, next_value_rec = gae_and_next_value
                done_batteries, values_batteries, rewards_batteries = (
                    transition.done_batteries,
                    transition.values_batteries,
                    transition.rewards_batteries,
                )
                done_rec, value_rec, reward_rec = (
                    transition.done_rec,
                    transition.value_rec,
                    transition.reward_rec,
                )

                delta_batteries = rewards_batteries + config["GAMMA"] * next_value_batteries * (1 - done_batteries) - values_batteries
                gae_batteries = (
                    delta_batteries
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done_batteries) * gae_batteries
                )
                delta_rec = reward_rec + config["GAMMA"] * next_value_rec * (1 - done_rec) - value_rec
                gae_rec = (
                        delta_rec
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done_rec) * gae_rec
                )
                return (gae_batteries, values_batteries, gae_rec, value_rec), (gae_batteries, gae_rec)

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val_batteries), last_val_batteries, jnp.zeros_like(last_val_rec), last_val_rec),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            advantages_batteries, advantages_rec = advantages


            return advantages, (advantages_batteries + traj_batch.values_batteries, advantages_rec + traj_batch.value_rec)

        advantages, targets = _calculate_gae(traj_batch, last_val_batteries, last_val_rec)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                # print(jax.tree.map(lambda x: x.shape, batch_info))
                advantages_batteries, advantages_rec = advantages
                targets_batteries, targets_rec = targets

                def _loss_fn(network, traj_batch_data, gae, targets):

                    traj_batch_obs, traj_batch_actions, traj_batch_values, traj_batch_log_probs = traj_batch_data

                    # RERUN NETWORK
                    pi, value = network(traj_batch_obs)
                    # print('davfvd')
                    log_prob = pi.log_prob(traj_batch_actions)
                    # print(log_prob.shape)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch_values + (
                        value - traj_batch_values
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch_log_probs)
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
                    return total_loss, (value_loss, loss_actor, entropy)        #fixme in realtà ogni agente ha la sua loss, ma così non dovrebbe cambiare

                networks_batteries, optimizer_batteries, network_rec, optimizer_rec = nnx.merge(train_state.graph_def, train_state.state)

                traj_data_batteries_for_loss = (traj_batch.obs_batteries, traj_batch.actions_batteries, traj_batch.values_batteries, traj_batch.log_prob_batteries)
                # num_agents axis must be first to work with network
                traj_data_batteries_for_loss = jax.tree.map(lambda x : x.swapaxes(0, 1), traj_data_batteries_for_loss)

                # print('traj_data_batteries_for_loss')
                # print(jax.tree_map(lambda x: x.shape, traj_data_batteries_for_loss))

                grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                total_loss_batteries, grads_batteries = grad_fn(
                    networks_batteries,
                    traj_data_batteries_for_loss,
                    advantages_batteries.swapaxes(0, 1),
                    targets_batteries.swapaxes(0, 1)
                )
                # print('tot loss batteites')
                # print(jax.tree_map(lambda x: x.shape, total_loss_batteries))

                traj_data_rec_for_loss = (traj_batch.obs_rec, traj_batch.actions_rec, traj_batch.value_rec, traj_batch.log_prob_rec)

                total_loss_rec, grads_rec = grad_fn(
                    network_rec,
                    traj_data_rec_for_loss,
                    advantages_rec,
                    targets_rec
                )

                optimizer_batteries.update(grads_batteries)
                # print('qui ci arrivo')
                optimizer_rec.update(grads_rec)
                train_state = train_state.replace(state=nnx.state((networks_batteries, optimizer_batteries, network_rec, optimizer_rec)))

                total_loss = (total_loss_batteries, total_loss_rec)

                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"]# * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)

            # print(jax.tree.map(lambda x: x.shape, batch))

            # jax.debug.print('bef {z}', z=jax.tree.map(lambda l: l.shape, traj_batch), ordered=True)

            # batch = jax.tree_util.tree_map(
            #     lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            # )
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
            # print(jax.tree.map(lambda x: x.shape, minibatches))
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

        runner_state = (train_state, env_state, last_obs_batteries, rng)

        runner_state_plus = (runner_state, val_info)

        return runner_state_plus, metric

    rng, _rng = jax.random.split(rng)
    obsv_batteries = jnp.vstack([obsv[a] for a in env.battery_agents])
    # print(obsv_batteries.shape)
    runner_state = (train_state, env_state, obsv_batteries, _rng)

    runner_state_plus = (runner_state, val_info)

    runner_state_plus, metric = jax.lax.scan(
        _update_step, runner_state_plus, jnp.arange(config["NUM_UPDATES"])
    )

    runner_state, val_info = runner_state_plus

    if validate:
        return {'runner_state': runner_state, 'metrics': metric, 'val_info': val_info}
    else:
        return {'runner_state': runner_state, 'metrics': metric}
