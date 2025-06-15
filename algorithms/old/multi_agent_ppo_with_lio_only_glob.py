from time import time
from datetime import datetime

import jax
import jax.numpy as jnp
from jax.experimental import io_callback

from flax import nnx
from flax import struct
from functools import partial

from algorithms.tqdm_custom import scan_tqdm as tqdm_custom

from flax.nnx import GraphDef, GraphState
import numpy as np
import optax
from typing import Sequence, NamedTuple, Any, Union

# from jaxmarl.wrappers.baselines import JaxMARLWrapper
from algorithms.wrappers import VecEnvJaxMARL

import algorithms.utils as utils
from ernestogym.envs.multi_agent.env_only_batteries import RECEnv, EnvState
from algorithms.networks import StackedActorCritic, StackedRecurrentActorCritic, RECActorCritic, RECRecurrentActorCritic
from algorithms.networks_lio import StackedIncentiveNetworkPercentage


class StackedOptimizer(nnx.Optimizer):

    def __init__(self, num_networks:int, models, tx):

        self.num_networks = num_networks

        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap(in_axes=(0, 0, None))
        def vmapped_fn(self, model, tx):
            super(StackedOptimizer, self).__init__(model, tx)

        vmapped_fn(self, models, tx)

    def update(self, grads, **kwargs):
        @nnx.split_rngs(splits=self.num_networks)
        @nnx.vmap
        def vmapped_fn(self, grads):
            super(StackedOptimizer, self).update(grads, **kwargs)

        vmapped_fn(self, grads)


def make_train(config, env:RECEnv):

    print('PPO NORMALE')

    config['NUM_BATTERY_AGENTS'] = env.num_battery_agents

    config['NUM_UPDATES'] = config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']
    config['MINIBATCH_SIZE'] = config['NUM_ENVS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES']

    config['BATTERY_ACTION_SPACE_SIZE'] = env.action_space(env.agents[0]).shape[0]
    config['BATTERY_OBS_KEYS'] = tuple(env.obs_keys)
    config['BATTERY_OBS_IS_SEQUENCE'] = env.obs_is_sequence
    config['BATTERY_OBS_IS_NORMALIZABLE'] = env.obs_is_normalizable

    assert (len(env.agents) ==
            config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS'] +
            config['NUM_ONLY_MARKET_AGENTS'] + config['NUM_RANDOM_AGENTS'])

    env = VecEnvJaxMARL(env)

    # env = LogWrapper(env)
    # env = ClipAction(env, low=env_params.i_min_action, high=env_params.i_max_action)
    # if config['NORMALIZE_ENV']:
    #     env = NormalizeVecObservation(env)
    #     env = NormalizeVecReward(env, config['GAMMA'])

    def schedule_builder(lr_init, lr_end, frac, tot_updates, warm_up):

        tot_steps = int(tot_updates * frac)
        warm_up_steps = int(tot_updates * warm_up)

        if config['LR_SCHEDULE'] == 'linear':
            return optax.schedules.linear_schedule(lr_init, lr_end, tot_steps)
        elif config['LR_SCHEDULE'] == 'cosine':
            optax.schedules.cosine_decay_schedule(lr_init, tot_steps, lr_end / lr_init)
            return optax.schedules.warmup_cosine_decay_schedule(0., lr_init, warm_up_steps, tot_steps, lr_end)
        else:
            return lr_init

    _rng = nnx.Rngs(123)
    networks_policy = utils.construct_battery_net_from_config_multi_agent(config, _rng, num_nets=config['NUM_RL_AGENTS'])

    config['PERCENTAGE_INCENTIVES'] = True
    _rng = nnx.Rngs(222)
    networks_incentives = utils.construct_incentive_net_from_config_multi_agent(config, _rng)

    schedule_policy = schedule_builder(config['LR_POLICY'], config['LR_POLICY_MIN'], config['FRACTION_DYNAMIC_LR_POLICY'],
                                       config['NUM_MINIBATCHES']*config['UPDATE_EPOCHS']*config['NUM_UPDATES'], warm_up=config.get('WARMUP_SCHEDULE_POLICY', 0))

    schedule_incentives = schedule_builder(config['LR_INCENTIVES'], config['LR_INCENTIVES_MIN'], config['FRACTION_DYNAMIC_LR_INCENTIVES'],
                                           config['NUM_UPDATES'], warm_up=config.get('WARMUP_SCHEDULE_INCENTIVES', 0))


    def get_optim(name, scheduler, beta=None):
        if beta is None:
            beta = 0.9
        if name == 'adam':
            return optax.adam(learning_rate=scheduler, eps=0., eps_root=1e-10, b1=beta)
        elif name == 'adamw':
            print('ADAMW')
            return optax.adamw(learning_rate=scheduler, eps=0., eps_root=1e-10, b1=beta)
        elif name == 'sgd':
            return optax.sgd(learning_rate=scheduler)
        elif name == 'rmsprop':
            return optax.rmsprop(learning_rate=scheduler, momentum=0.9)
        else:
            raise ValueError("Optimizer '{}' not recognized".format(name))

    tx_pol = optax.chain(
        optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
        get_optim(config['OPTIMIZER_POLICY'], schedule_policy, config.get('BETA_ADAM_POLICY', None))
    )
    tx_inc = get_optim(config['OPTIMIZER_INCENTIVES'], schedule_incentives, config.get('BETA_ADAM_INCENTIVES', None))


    optimizers_policy = StackedOptimizer(config['NUM_RL_AGENTS'], networks_policy, tx_pol)
    optimizers_incentives = StackedOptimizer(config['NUM_RL_AGENTS'], networks_incentives, tx_inc)
    # optimizers_incentives = StackedOptimizer(config['NUM_RL_AGENTS'], networks_incentives, optax.apply_if_finite(tx_inc, 2))

    graph_def, state = nnx.split((networks_policy, optimizers_policy, networks_incentives, optimizers_incentives))

    train_state = TrainState(graph_def=graph_def, state=state)

    return env, train_state

@struct.dataclass
class TrainState:
    graph_def: GraphDef
    state: GraphState

class LSTMState(NamedTuple):
    act_state: tuple = ()
    cri_state: tuple = ()

class RunnerState(NamedTuple):
    # train_state: TrainState
    networks_policy: Union[StackedActorCritic, StackedRecurrentActorCritic]
    networks_incentives: StackedIncentiveNetworkPercentage
    optimizers_policy: StackedOptimizer
    optimizers_incentives: nnx.Optimizer

    env_state: EnvState
    last_obs: jnp.ndarray
    rng: jax.random.PRNGKey
    done_prev: jnp.ndarray = jnp.array(False)
    last_lstm_state: LSTMState = LSTMState()


class Transition(NamedTuple):
    done: jnp.ndarray
    actions: jnp.ndarray
    values: jnp.ndarray
    ext_rewards: jnp.ndarray
    # int_rewards: jnp.ndarray
    # tot_rewards: jnp.ndarray
    rewards: jnp.ndarray
    int_rewards_mat: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

    done_prev: jnp.ndarray = 0
    lstm_states_prev: LSTMState = LSTMState((), ())


def train_wrapper(env:RECEnv, config, networks_policy, optimizers_policy, networks_incentives, optimizers_incentives,
                  rng, world_metadata, rec_rule_based_policy=None, validate=True, freq_val=None, val_env=None, val_rng=None,
                  val_num_iters=None, path_saving=None):

    infos = {}
    val_infos = {}

    dir_name = (datetime.now().strftime('%Y%m%d_%H%M%S') +
                '_pol_net_type_' + str(config['NETWORK_TYPE_BATTERIES']) +
                '_lr_pol_' + str(config.get('LR_POLICY')) +
                '_lr_inc_' + str(config.get('LR_INCENTIVES')) +
                '_tot_timesteps_' + str(config.get('TOTAL_TIMESTEPS')) +
                '_lr_sched_' + str(config.get('LR_SCHEDULE')) +
                '_multiagent' + '/')

    directory = path_saving + dir_name

    def end_update_step(info, i):
        if len(infos) == 0:
            # infos = jax.tree.map(lambda x: np.empty_like(x, shape=(config['NUM_UPDATES'],)+x.shape), info)
            infos.update(jax.tree.map(lambda x: np.empty_like(x, shape=(config['NUM_UPDATES'],) + x.shape), info))

        info = jax.device_put(info, device=jax.devices('cpu')[0])

        def update(logs, new):
            logs[i] = new

        jax.tree.map(update, infos, info)

    def update_val_info(val_info, train_state, i):
        if len(val_infos) == 0:
            # infos = jax.tree.map(lambda x: np.empty_like(x, shape=(config['NUM_UPDATES'],)+x.shape), info)
            val_infos.update(
                jax.tree.map(lambda x: np.empty_like(x, shape=((config['NUM_UPDATES'] - 1) // freq_val + 1,) + x.shape),
                             val_info))

        def update(logs, new):
            logs[i] = new

        networks_policy, networks_incentives = nnx.merge(train_state.graph_def, train_state.state)

        val_info = jax.device_put(val_info, device=jax.devices('cpu')[0])
        jax.tree.map(update, val_infos, val_info)

        utils.save_state_multiagent_only_batteries(directory, networks_policy, networks_incentives, config, world_metadata, is_checkpoint=True, num_steps=i)

    @partial(nnx.jit, static_argnums=(0, 1, 7, 8, 9, 11))
    def train(env: RECEnv, config, network_policy, optimizer_policy, network_incentives, optimizer_incentives, rng,
              validate=True, freq_val=None, val_env=None,val_rng=None, val_num_iters=None):

        if validate:
            if freq_val is None or val_env is None or val_rng is None or val_num_iters is None:
                raise ValueError(
                    "'freq_val', 'val_env', 'val_rng' and 'val_num_iters' must be defined when 'validate' is True")

        rng, _rng = jax.random.split(rng)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config['NUM_ENVS'])
        obsv, env_state = env.reset(reset_rng)

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic' and config['NUM_RL_AGENTS'] > 0:
            act_state_batteries, cri_state_batteries = network_policy.get_initial_lstm_state()
            act_state_batteries, cri_state_batteries = jax.tree.map(
                lambda x: jnp.tile(x[None, :], (config['NUM_ENVS'],) + (1,) * len(x.shape)),
                (act_state_batteries, cri_state_batteries))
            lstm_state_batteries = LSTMState(act_state_batteries, cri_state_batteries)

            episode_starts_batteries = jnp.ones((config['NUM_ENVS'], config['NUM_RL_AGENTS']), dtype=bool)
        else:
            lstm_state_batteries = LSTMState((jnp.ones(config['NUM_ENVS'], dtype=bool),),
                                             (jnp.ones(config['NUM_ENVS'], dtype=bool),))  # dummy
            episode_starts_batteries = jnp.ones(config['NUM_ENVS'], dtype=bool)  # dummy

        # TRAIN LOOP
        # @scan_tqdm(config['NUM_UPDATES'], print_rate=1)
        @tqdm_custom(0, 0, 1, config['NUM_UPDATES'], print_rate=1)
        def _update_step(runner_state:RunnerState, curr_iter):
            # COLLECT TRAJECTORIES

            runner_state = update_incentive_network(runner_state, env, config)

            if validate:
                _ = jax.lax.cond(curr_iter % freq_val == 0,
                                 lambda: io_callback(update_val_info,
                                                     None,
                                                     test_networks(val_env, TrainState(*nnx.split((runner_state.networks_policy, runner_state.networks_incentives))),
                                                                   val_num_iters, config, val_rng, rec_rule_based_policy=rec_rule_based_policy,
                                                                   curr_iter=curr_iter, print_data=True),
                                                     TrainState(*nnx.split((runner_state.networks_policy, runner_state.networks_incentives))),
                                                     curr_iter // freq_val,
                                                     ordered=True),
                                 lambda: None)

            # if config.get('SAVE_TRAIN_INFO', False):
            #     io_callback(end_update_step, None, metric, curr_iter, ordered=True)

            return runner_state

        rng, _rng = jax.random.split(rng)
        obsv_batteries = jax.tree.map(lambda *vals: jnp.stack(vals, axis=1), *[obsv[a] for a in env.agents])

        if config.get('REC_VALUE_IN_BATTERY_OBS', False) or config.get('REC_VALUE_IN_BATTERY_OBS_CRI', False):
            obsv_batteries['rec_value'] = jnp.zeros((config['NUM_ENVS'], config['NUM_BATTERY_AGENTS']))

        network_policy.eval()
        network_incentives.eval()

        runner_state = RunnerState(networks_policy=network_policy,
                                   networks_incentives=network_incentives,
                                   optimizers_policy=optimizer_policy,
                                   optimizers_incentives=optimizer_incentives,
                                   env_state=env_state,
                                   last_obs=obsv_batteries,
                                   rng=_rng,
                                   done_prev=episode_starts_batteries,
                                   last_lstm_state=lstm_state_batteries)

        scanned_update_step = nnx.scan(_update_step,
                                       in_axes=(nnx.Carry, 0),
                                       out_axes=nnx.Carry)

        runner_state = scanned_update_step(runner_state, jnp.arange(config['NUM_UPDATES']))

        runner_state.networks_policy.eval()
        runner_state.networks_incentives.eval()

        return runner_state

    runner_state = train(env, config, networks_policy, optimizers_policy, networks_incentives, optimizers_incentives,
                         rng, validate, freq_val, val_env, val_rng, val_num_iters)

    print('Saving...')

    t0 = time()

    utils.save_state_multiagent_only_batteries(directory, runner_state.networks_policy, runner_state.networks_incentives, config, world_metadata, infos, val_infos, is_checkpoint=False)

    # metric = jax.tree.map(lambda *vals: np.stack(vals, axis=0), *infos)

    print(f'Saving time: {t0-time():.2f} s')

    if validate:
        # val_info = jax.tree.map(lambda *vals: np.stack(vals, axis=0), *val_infos)
        return {'runner_state': runner_state, 'metrics': infos, 'val_info': val_infos}
    else:
        return {'runner_state': runner_state, 'metrics': infos}


# def get_others_actions(actions):
#     others_actions = []
#     others_actions.append(actions[..., 1:])
#     for i in range(1, actions.shape[-1] - 1):
#         others_actions.append(jnp.concat((actions[..., :i], actions[..., i+1:]), axis=-1))
#     others_actions.append(actions[..., :-1])
#
#     return jnp.stack(others_actions, axis=0)
#
# def add_zero_diag(int_rew):
#     def _add_zero_diag(mat):
#         n, m = mat.shape
#         full = jnp.zeros((n, m+1))
#
#         ind = np.logical_not(np.eye(n, m+1, dtype=bool))
#
#         # Flatten x and scatter into full matrix
#         full = full.at[ind].set(mat.flatten())
#
#         return full
#
#     if int_rew.ndim == 2:
#         return _add_zero_diag(int_rew)
#     elif int_rew.ndim > 2:
#         return jax.vmap(add_zero_diag, in_axes=0)(int_rew)
#     else:
#         raise ValueError('int_rew must be 2 or more')


def collect_trajectories(runner_state: RunnerState, config, env):

    def _env_step(runner_state: RunnerState):

        network_policy, network_incentives = runner_state.networks_policy, runner_state.networks_incentives

        # SELECT ACTION
        rng, _rng = jax.random.split(runner_state.rng)

        actions = []

        last_obs_rl_num_batteries_first = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1)[:config['NUM_RL_AGENTS']],
                                                       runner_state.last_obs)

        if config['NUM_RL_AGENTS'] > 0:
            if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
                prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: jnp.where(runner_state.done_prev[(slice(None), slice(None)) + (None,)*(x.ndim-1)], x[None, :], y),
                                                              network_policy.get_initial_lstm_state(),
                                                              (runner_state.last_lstm_state.act_state, runner_state.last_lstm_state.cri_state))
                prev_act_state_num_batteries_first, prev_cri_state_num_batteries_first = jax.tree.map(lambda x : jnp.swapaxes(x, 0, 1), (prev_act_state, prev_cri_state))
                pi, values, lstm_act_state, lstm_cri_state = network_policy(last_obs_rl_num_batteries_first, prev_act_state_num_batteries_first, prev_cri_state_num_batteries_first)
                lstm_act_states, lstm_cri_states = jax.tree.map(lambda x : jnp.swapaxes(x, 0, 1), (lstm_act_state, lstm_cri_state))
            else:
                pi, values = network_policy(last_obs_rl_num_batteries_first)
                lstm_act_states, lstm_cri_states = runner_state.last_lstm_state.act_state, runner_state.last_lstm_state.cri_state

            # _rng = jax.lax.stop_gradient(_rng)

            actions_rl = pi.sample(seed=_rng)                        # batteries first
            # actions_rl = pi.mode()
            log_probs = pi.log_prob(actions_rl)             # batteries first

            values, actions_rl, log_probs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1),
                                                                                  (values, actions_rl,
                                                                                   log_probs))  # num_envs first

            actions.append(actions_rl)
        else:
            values = jnp.zeros((config['NUM_ENVS'],))
            log_probs = jnp.zeros((config['NUM_ENVS'],))
            lstm_act_states, lstm_cri_states = runner_state.last_lstm_state.act_state, runner_state.last_lstm_state.cri_state


        if config['NUM_BATTERY_FIRST_AGENTS'] > 0:
            idx_start_bf = config['NUM_RL_AGENTS']
            idx_end_bf = config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS']

            demand = runner_state.last_obs['demand'][:, idx_start_bf:idx_end_bf]
            generation = runner_state.last_obs['generation'][:, idx_start_bf:idx_end_bf]

            actions_battery_first = (generation - demand) / runner_state.env_state.battery_states.electrical_state.v[:, idx_start_bf:idx_end_bf]

            actions_battery_first = jnp.expand_dims(actions_battery_first, -1)

            actions.append(actions_battery_first)

        if config['NUM_ONLY_MARKET_AGENTS'] > 0:
            actions_only_market = jnp.zeros((config['NUM_ENVS'], config['NUM_ONLY_MARKET_AGENTS'], config['BATTERY_ACTION_SPACE_SIZE']))
            actions.append(actions_only_market)

        if config['NUM_RANDOM_AGENTS'] > 0:
            rng, _rng = jax.random.split(rng)

            actions_random = jax.random.uniform(_rng, shape=(config['NUM_ENVS'], config['NUM_RANDOM_AGENTS']),
                                                minval=-1., maxval=1.)

            actions_random *= config['MAX_ACTION_RANDOM_AGENTS']

            actions_random = jnp.expand_dims(actions_random, -1)

            actions.append(actions_random)

        actions = jnp.concat(actions, axis=1)

        actions_dict = {env.agents[i]: actions[:, i] for i in range(env.num_battery_agents)}

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config['NUM_ENVS'])
        env_state = runner_state.env_state
        obsv, env_state, reward_env, done, info = env.step(rng_step, env_state, actions_dict)

        obsv = jax.lax.stop_gradient(obsv)
        env_state = jax.lax.stop_gradient(env_state)
        done = jax.lax.stop_gradient(done)
        info = jax.lax.stop_gradient(info)

        # reward_env = jnp.stack([reward_env[a] for a in env.agents], axis=1)

        reward_env = jax.tree.map(lambda *vals: jnp.stack(vals, axis=1), *[reward_env[a] for a in env.agents])
        reward_env = {key: (val if key == 'r_glob' else jax.lax.stop_gradient(val)) for key, val in reward_env.items()}

        broadcast_actions = jnp.stack([actions.squeeze(axis=-1)]*config['NUM_RL_AGENTS'], axis=0)

        info['actions'] = actions_dict

        # others_actions = get_others_actions(jnp.squeeze(actions, axis=-1))

        int_rewards_separate_frac_num_batteries_first = network_incentives(last_obs_rl_num_batteries_first, broadcast_actions)
        int_rewards_separate_frac = jnp.swapaxes(int_rewards_separate_frac_num_batteries_first, 0, 1)

        # jax.debug.print('{x}', x=int_rewards_separate_frac, ordered=True)

        if config['NUM_RL_AGENTS'] < config['NUM_BATTERY_AGENTS']:
            n_ag = config['NUM_BATTERY_AGENTS']
            n_rl = config['NUM_RL_AGENTS']
            int_rewards_separate_frac_rule_based = jnp.stack([jnp.eye(n_ag-n_rl, n_ag, n_rl)]*config['NUM_ENVS'], axis=0)
            int_rewards_separate_frac = jnp.concat([int_rewards_separate_frac, int_rewards_separate_frac_rule_based], axis=1)

        int_rewards_mat = int_rewards_separate_frac * reward_env['r_glob'][..., None]

        # jax.debug.print('rew mat {x}', x=int_rewards_mat)
        int_rewards = int_rewards_mat.sum(axis=-2)

        reward = jnp.stack([val for key, val in reward_env.items() if key != 'r_glob'], axis=0).sum(axis=0) + int_rewards

        info['int_rewards'] = int_rewards

        done = jnp.stack([done[a] for a in env.agents], axis=1)

        obsv = jax.tree.map(lambda *vals: jnp.stack(vals, axis=1), *[obsv[a] for a in env.agents])

        transition = Transition(
            done=done,
            actions=actions,
            values=values,
            ext_rewards=reward_env,
            rewards=reward,
            # int_rewards=int_rewards,
            # tot_rewards=reward_env+int_rewards,
            int_rewards_mat=int_rewards_mat,
            log_prob=log_probs,
            obs=runner_state.last_obs,
            info=info,
            done_prev=runner_state.done_prev,
            lstm_states_prev=runner_state.last_lstm_state
        )

        new_done_prev = done[:, :config['NUM_RL_AGENTS']] if (config['NUM_RL_AGENTS']>0 and config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic') else runner_state.done_prev


        # jax.debug.print('{t}', t=transition, ordered=True)
        runner_state = runner_state._replace(env_state=env_state,
                                             last_obs=obsv,
                                             rng=rng,
                                             done_prev=new_done_prev,
                                             last_lstm_state=LSTMState(lstm_act_states,
                                                                       lstm_cri_states))

        return runner_state, transition

    # jax.debug.print('collecting traj', ordered=True)

    # runner_state, traj_batch = jax.lax.scan(
    #     _env_step, runner_state, None, config['NUM_STEPS']
    # )

    runner_state.networks_policy.eval()
    runner_state.networks_incentives.eval()

    runner_state, traj_batch = nnx.scan(_env_step,
                                        in_axes=nnx.Carry,
                                        out_axes=(nnx.Carry, 0),
                                        length=config['NUM_STEPS'])(runner_state)

    _, transition = _env_step(runner_state)

    last_val_batteries = transition.values

    return runner_state, traj_batch, last_val_batteries


def _calculate_gae(traj_batch:Transition, last_val, config):

    def _get_advantages(gae_and_next_value, transition_data):
        gae, next_value = gae_and_next_value
        done, value, rewards = transition_data

        # delta = rewards + config['GAMMA'] * next_value * (1 - done) - value
        # gae = (delta + config['GAMMA'] * config['GAE_LAMBDA'] * (1 - done) * gae)

        delta = rewards + config['GAMMA'] * next_value - value
        gae = (delta + config['GAMMA'] * config['GAE_LAMBDA'] * gae)

        return (gae, value), gae

    tot_rewards = traj_batch.rewards[..., :config['NUM_RL_AGENTS']] # ext_reward[..., :config['NUM_RL_AGENTS']] + int_reward[..., :config['NUM_RL_AGENTS']]

    if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS']:
        tot_rewards = (tot_rewards - tot_rewards.mean(axis=(0, 1), keepdims=True)) / (tot_rewards.std(axis=(0, 1), keepdims=True) + 1e-8)

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        (traj_batch.done[..., :config['NUM_RL_AGENTS']], traj_batch.values[..., :config['NUM_RL_AGENTS']], tot_rewards),
        reverse=True,
        unroll=32,
    )
    targets = advantages + traj_batch.values[..., :config['NUM_RL_AGENTS']]


    return advantages, targets

def update_incentive_network(runner_state:RunnerState, env, config):

    def inc_loss(net_inc, runner_state:RunnerState):

        runner_state = runner_state._replace(networks_incentives=net_inc)

        def loss_first_term(traj_batch_second:Transition, last_val):
            def compute_return(cumul, rew):
                cumul *= config['GAMMA']
                cumul += rew
                return cumul, cumul

            _, returns = jax.lax.scan(compute_return,
                                      last_val[..., :config['NUM_RL_AGENTS']],
                                      # jnp.zeros((config['NUM_ENVS'], config['NUM_RL_AGENTS'])),
                                      traj_batch_second.rewards[..., :config['NUM_RL_AGENTS']], #FIXME it was ext_reward before
                                      reverse=True,
                                      unroll=32)

            assert returns.shape == (config['NUM_STEPS'], config['NUM_ENVS'], config['NUM_RL_AGENTS'])

            # loss = jnp.expand_dims(traj_batch_second.log_prob, axis=2) * jnp.expand_dims(returns, axis=3)
            loss = traj_batch_second.log_prob[:, :, None, :] * returns[:, :, :, None]

            # jax.debug.print('log_prob: {x}\n\nreturns: {y}\n\n', x=traj_batch_second.log_prob, y=returns, ordered=True)

            assert loss.shape == (config['NUM_STEPS'], config['NUM_ENVS'], config['NUM_RL_AGENTS'], config['NUM_RL_AGENTS'])

            loss = loss.mean(axis=1)
            loss = loss.sum(axis=0)
            if config.get('ONLY_OTHERS', True):
                print('no 0 diagonal')
                loss = jnp.fill_diagonal(loss, 0., inplace=False)
            return - loss.sum(axis=1)

        def loss_second_term(int_rewards_mat):
            gammas = config['GAMMA'] ** jnp.arange(config['NUM_STEPS'])

            # int_rewards_mat_to_others = jnp.fill_diagonal(int_rewards_mat, 0., inplace=False)
            int_rewards_mat_rl = int_rewards_mat[:, :, :config['NUM_RL_AGENTS']]
            int_rewards_mat_to_others = int_rewards_mat_rl * (1 - jnp.eye(*int_rewards_mat_rl.shape[-2:]))

            # jax.debug.print('{x}', x=(int_rewards_mat_to_others >= 0.).all())

            norms = int_rewards_mat_to_others.sum(axis=-1)

            #fixme just to try
            # loss = gammas[:, None, None] * norms
            loss = norms

            loss = loss.mean(axis=1)

            return loss.sum(axis=0)

        # runner_state, traj_batch_first, last_val_batteries = jax.lax.stop_gradient(collect_trajectories(runner_state, config, env))
        runner_state.networks_incentives.train()
        runner_state, traj_batch_first, last_val_batteries = collect_trajectories(runner_state, config, env)
        runner_state.networks_incentives.eval()

        advantages, targets = _calculate_gae(traj_batch_first,
                                             last_val_batteries,
                                             config)

        # UPDATE NETWORKS
        # rng_upd = runner_state.rng
        runner_state, total_loss_batteries = update_policy_network(runner_state, traj_batch_first,
                                                                   advantages, targets,
                                                                   config)

        # runner_state, traj_batch_second, last_val_batteries = jax.lax.stop_gradient(collect_trajectories(runner_state, config, env))
        runner_state, traj_batch_second, last_val_batteries = collect_trajectories(runner_state, config, env)

        loss_first = loss_first_term(traj_batch_second, last_val_batteries)
        loss_second = loss_second_term(traj_batch_first.int_rewards_mat)

        # jax.debug.print('rew mat {x}', x=traj_batch_first.int_rewards_mat.sum(axis=0).mean(axis=0))

        # jax.debug.print('type target {x}', x=targets.dtype, ordered=True)
        #
        # jax.debug.print('loss first {x}', x=loss_first, ordered=True)
        # jax.debug.print('loss second {x}', x=loss_second, ordered=True)

        #fixme debug
        loss = loss_first + config['ALPHA'] * loss_second
        # loss = loss_second

        jax.debug.print('main loss: {x}, norm loss: {y}', x=loss_first, y=config['ALPHA']*loss_second)
        # loss = loss_first
        loss = loss.sum()
        runner_state.networks_incentives.eval()

        return loss, runner_state



    networks_incentives = runner_state.networks_incentives
    opt_inc = runner_state.optimizers_incentives
    # networks_incentives.train()
    
    grad_fn = nnx.grad(inc_loss, has_aux=True)

    runner_state = runner_state._replace(networks_incentives=None, optimizers_incentives=None)

    grads, runner_state = grad_fn(networks_incentives, runner_state)

    runner_state = runner_state._replace(optimizers_incentives=opt_inc)

    # jax.debug.print('grad inc {x}', x=grads.layers[0].bias.value, ordered=True)

    runner_state.optimizers_incentives.update(grads)

    # jax.debug.print('net {x}', x=runner_state.networks_incentives.layers[0].bias.value, ordered=True)
    # jax.debug.print('net2 {x}', x=networks_incentives.layers[0].bias.value, ordered=True)

    
    return runner_state



class UpdateState(NamedTuple):
    network: Union[StackedActorCritic, StackedRecurrentActorCritic, RECActorCritic, RECRecurrentActorCritic]
    optimizer: Union[StackedOptimizer, nnx.Optimizer]
    traj_batch: dict
    advantages: jnp.array
    targets: jnp.array
    rng: jax.random.PRNGKey

# def update_policy_network(runner_state: RunnerState, traj_batch, advantages, targets, config):
#     def _update_epoch(update_state: UpdateState):
#         def _update_minbatch(net_and_optim, traj_batch, advantages, targets):
#             network_policy, optimizer_policy = net_and_optim
#
#             # print(jax.tree.map(lambda x: x.shape, batch_info))
#
#             def _loss_fn_batteries(network, traj_batch, gae, targets):
#                 traj_batch_data = (traj_batch.obs,
#                                    traj_batch.actions,
#                                    traj_batch.values,
#                                    traj_batch.log_prob)
#                 traj_batch_obs, traj_batch_actions, traj_batch_values, traj_batch_log_probs = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_batch_data)
#                 traj_batch_obs, traj_batch_actions = jax.tree.map(lambda x: x[:config['NUM_RL_AGENTS']], (traj_batch_obs, traj_batch_actions))
#
#                 gae, targets = jax.tree.map(lambda x: x.swapaxes(0, 1), (gae, targets))
#
#                 # RERUN NETWORK
#                 pi, value = network(traj_batch_obs)
#                 log_prob = pi.log_prob(traj_batch_actions)
#                 # print(log_prob.shape)
#
#                 if config['NORMALIZE_TARGETS']:
#                     targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)
#
#                 # CALCULATE VALUE LOSS
#                 value_pred_clipped = traj_batch_values + (
#                         value - traj_batch_values
#                 ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
#                 value_losses = jnp.square(value - targets)
#                 value_losses_clipped = jnp.square(value_pred_clipped - targets)
#                 # jax.debug.print('value losses {x}', x=value_losses.shape)
#                 value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=1)
#
#                 #fixme debug
#                 # value_loss = jnp.zeros_like(value_loss)
#                 # value_loss = jax.lax.stop_gradient(value_loss)
#
#                 assert value_loss.shape == (config['NUM_RL_AGENTS'],)
#
#                 # CALCULATE ACTOR LOSS
#                 ratio = jnp.exp(log_prob - traj_batch_log_probs)
#
#                 # jax.debug.print('ratio mean {x}, max {y}, min {z}, std {w}', x=ratio.mean(), y=ratio.max(), z=ratio.min(), w=ratio.std(), ordered=True)
#
#                 if config['NORMALIZE_ADVANTAGES']:
#                     # jax.debug.print('gae before {x}', x=gae, ordered=True)
#                     gae = (gae - gae.mean(axis=1, keepdims=True)) / (gae.std(axis=1, keepdims=True) + 1e-8)
#                     # jax.debug.print('gae after {x}', x=gae, ordered = True)
#
#                 loss_actor1 = ratio * gae
#                 # jax.debug.print('loss_actor1 {x}', x=loss_actor1, ordered=True)
#                 loss_actor2 = (
#                         jnp.clip(
#                             ratio,
#                             1.0 - config['CLIP_EPS'],
#                             1.0 + config['CLIP_EPS'],
#                         )
#                         * gae
#                 )
#                 # jax.debug.print('loss_actor2 {x}', x=loss_actor2, ordered=True)
#                 # jax.debug.print('means {x}, {y}', x=loss_actor1.mean(), y=loss_actor2.mean(), ordered=True)
#                 loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
#                 loss_actor = loss_actor.mean(axis=1)
#
#                 #fixme debug
#                 # loss_actor = jax.lax.stop_gradient(loss_actor)
#
#                 entropy = pi.entropy()
#                 entropy = entropy.mean(axis=1)
#
#                 total_loss = (
#                         loss_actor
#                         + config['VF_COEF'] * value_loss
#                         - config['ENT_COEF_BATTERIES'] * entropy
#                 )
#
#                 assert total_loss.ndim == 1
#                 # jax.debug.print('bat_loss', ordered=True)
#
#                 total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
#                 return total_loss, (value_loss, loss_actor, entropy)
#
#             grad_fn_batteries = nnx.value_and_grad(_loss_fn_batteries, has_aux=True)
#
#             # traj_batch = jax.lax.stop_gradient(traj_batch)
#
#             total_loss_batteries, grads_batteries = grad_fn_batteries(
#                 network_policy,
#                 traj_batch,
#                 advantages,
#                 targets
#             )
#
#             optimizer_policy.update(grads_batteries)
#
#             total_loss = total_loss_batteries
#
#             return (network_policy, optimizer_policy), total_loss
#
#         # train_state, traj_batch, advantages, targets, rng = update_state
#
#         batch_size = config['MINIBATCH_SIZE'] * config['NUM_MINIBATCHES']
#         assert (
#                 batch_size == config['NUM_STEPS'] * config['NUM_ENVS']
#         ), 'batch size must be equal to number of steps * number of envs'
#         print('adv', advantages.shape)
#
#         batch = (traj_batch, advantages, targets)
#
#         batch = jax.tree.map(
#             lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
#         )
#
#         _, total_loss = _update_minbatch((update_state.network, update_state.optimizer), *batch)
#
#         return update_state, total_loss
#
#     runner_state.networks_policy.train()
#
#     update_state = UpdateState(network=runner_state.networks_policy, optimizer=runner_state.optimizers_policy,
#                                traj_batch=traj_batch, advantages=advantages, targets=targets, rng=runner_state.rng)
#
#     update_state, loss_info = _update_epoch(update_state)
#
#
#     # update_state, loss_info = jax.lax.scan(
#     #     _update_epoch, update_state, None, config['UPDATE_EPOCHS']
#     # )
#
#     # jax.debug.print('bat loss {x}', x=loss_info[0])
#
#     runner_state.networks_policy.eval()
#
#     runner_state = runner_state._replace(rng=update_state.rng)
#
#     return runner_state, loss_info


def update_policy_network(runner_state: RunnerState, traj_batch, advantages, targets, config):
    def _update_epoch(update_state: UpdateState):
        def _update_minbatch(net_and_optim, traj_batch, advantages, targets):
            network_policy, optimizer_policy = net_and_optim

            # print(jax.tree.map(lambda x: x.shape, batch_info))

            def _loss_fn_batteries(network, traj_batch, gae, targets):
                traj_batch_data = (traj_batch.obs,
                                   traj_batch.actions,
                                   traj_batch.values,
                                   traj_batch.log_prob)
                traj_batch_obs, traj_batch_actions, traj_batch_values, traj_batch_log_probs = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_batch_data)
                traj_batch_obs, traj_batch_actions = jax.tree.map(lambda x: x[:config['NUM_RL_AGENTS']], (traj_batch_obs, traj_batch_actions))

                gae, targets = jax.tree.map(lambda x: x.swapaxes(0, 1), (gae, targets))

                # RERUN NETWORK
                pi, value = network(traj_batch_obs)
                log_prob = pi.log_prob(traj_batch_actions)
                # print(log_prob.shape)

                if config['NORMALIZE_TARGETS']:
                    targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch_values + (
                        value - traj_batch_values
                ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                # jax.debug.print('value losses {x}', x=value_losses.shape)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=1)

                #fixme debug
                # value_loss = jnp.zeros_like(value_loss)
                # value_loss = jax.lax.stop_gradient(value_loss)

                assert value_loss.shape == (config['NUM_RL_AGENTS'],)

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch_log_probs)

                # jax.debug.print('ratio mean {x}, max {y}, min {z}, std {w}', x=ratio.mean(), y=ratio.max(), z=ratio.min(), w=ratio.std(), ordered=True)

                if config['NORMALIZE_ADVANTAGES']:
                    # jax.debug.print('gae before {x}', x=gae, ordered=True)
                    gae = (gae - gae.mean(axis=1, keepdims=True)) / (gae.std(axis=1, keepdims=True) + 1e-8)
                    # jax.debug.print('gae after {x}', x=gae, ordered = True)

                loss_actor1 = ratio * gae
                # jax.debug.print('loss_actor1 {x}', x=loss_actor1, ordered=True)
                loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config['CLIP_EPS'],
                            1.0 + config['CLIP_EPS'],
                        )
                        * gae
                )
                # jax.debug.print('loss_actor2 {x}', x=loss_actor2, ordered=True)
                # jax.debug.print('means {x}, {y}', x=loss_actor1.mean(), y=loss_actor2.mean(), ordered=True)
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean(axis=1)

                #fixme debug
                # loss_actor = jax.lax.stop_gradient(loss_actor)

                entropy = pi.entropy()
                entropy = entropy.mean(axis=1)

                total_loss = (
                        loss_actor
                        + config['VF_COEF'] * value_loss
                        - config['ENT_COEF_BATTERIES'] * entropy
                )

                assert total_loss.ndim == 1
                # jax.debug.print('bat_loss', ordered=True)

                total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
                return total_loss, (value_loss, loss_actor, entropy)

            def _loss_fn_batteries_recurrent(network, traj_batch:Transition, gae, targets):

                obs_battery_first = jax.tree.map(lambda x : x.swapaxes(0, 1)[:config['NUM_RL_AGENTS']], traj_batch.obs)
                # obs_battery_first = traj_batch.obs_batteries[:, :config['NUM_RL_AGENTS']].swapaxes(0, 1)

                data_for_network_act, data_for_network_cri = network.prepare_data(obs_battery_first, return_cri=True)

                def forward_pass_lstm(carry, data_act, data_cri, beginning):
                    network, act_state, cri_state = carry

                    act_state, cri_state = jax.tree.map(lambda x, y: jnp.where(beginning[(slice(None),) + (None,)*(x.ndim-1)], x, y),
                                                        network.get_initial_lstm_state(),
                                                        (act_state, cri_state))
                    #jnp.where(beginning, init_states, states)
                    act_state, act_output = network.apply_lstm_act(data_act, act_state)
                    cri_state, cri_output = network.apply_lstm_cri(data_cri, cri_state)
                    return (network, act_state, cri_state), act_output, cri_output

                _, act_outputs, cri_outputs = nnx.scan(forward_pass_lstm,
                                                       in_axes=(nnx.Carry, 1, 1, 0),
                                                       out_axes=(nnx.Carry, 1, 1),
                                                       unroll=16)((network,) + jax.tree.map(lambda x: x[0], (traj_batch.lstm_states_prev.act_state, traj_batch.lstm_states_prev.cri_state)),
                                                                  data_for_network_act, data_for_network_cri, traj_batch.done_prev[:, :config['NUM_RL_AGENTS']])

                pi = network.apply_act_mlp(data_for_network_act, act_outputs)
                values = network.apply_cri_mlp(data_for_network_cri, cri_outputs)
                log_prob = pi.log_prob(traj_batch.actions[:, :config['NUM_RL_AGENTS']].swapaxes(0, 1))

                values_time_first = values.swapaxes(0, 1)
                log_prob_time_first = log_prob.swapaxes(0, 1)

                if config['NORMALIZE_TARGETS']:
                    targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.values + (
                        values_time_first - traj_batch.values
                ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                value_losses = jnp.square(values_time_first - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                # jax.debug.print('value losses {x}', x=value_losses.shape)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=0)

                assert value_loss.shape == (config['NUM_RL_AGENTS'],)

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob_time_first - traj_batch.log_prob)

                if config['NORMALIZE_ADVANTAGES']:
                    gae = (gae - gae.mean(axis=0, keepdims=True)) / (gae.std(axis=0, keepdims=True) + 1e-8)

                loss_actor1 = ratio * gae
                loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config['CLIP_EPS'],
                            1.0 + config['CLIP_EPS'],
                        )
                        * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean(axis=0)    # time first
                entropy = pi.entropy()
                entropy = entropy.mean(axis=1)          #batteries first

                total_loss = (
                        loss_actor
                        + config['VF_COEF'] * value_loss
                        - config['ENT_COEF_BATTERIES'] * entropy
                )

                assert total_loss.ndim == 1

                total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
                return total_loss, (value_loss, loss_actor, entropy)

            # traj_data_batteries_for_loss = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_data_batteries_for_loss)

            # print('traj_data_batteries_for_loss')
            # print(jax.tree.map(lambda x: x.shape, traj_data_batteries_for_loss))
            if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
                grad_fn_batteries = nnx.value_and_grad(_loss_fn_batteries_recurrent, has_aux=True)
                # traj_data_batteries_for_loss += (traj_batch.done_prev_batteries,)
            else:
                grad_fn_batteries = nnx.value_and_grad(_loss_fn_batteries, has_aux=True)

            # traj_batch = jax.lax.stop_gradient(traj_batch)

            total_loss_batteries, grads_batteries = grad_fn_batteries(
                network_policy,
                traj_batch,
                advantages,
                targets
            )

            # grads_batteries = jax.lax.stop_gradient(grads_batteries)

            optimizer_policy.update(grads_batteries)

            total_loss = total_loss_batteries

            return (network_policy, optimizer_policy), total_loss

        # train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(update_state.rng)
        batch_size = config['MINIBATCH_SIZE'] * config['NUM_MINIBATCHES']
        assert (
                batch_size == config['NUM_STEPS'] * config['NUM_ENVS']
        ), 'batch size must be equal to number of steps * number of envs'

        batch = (traj_batch, advantages, targets)

        # print(jax.tree.map(lambda x: x.shape, batch))

        # jax.debug.print('bef {z}', z=jax.tree.map(lambda l: l.shape, traj_batch), ordered=True)

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1), batch
            )
            batch = jax.tree.map(
                lambda x: x.reshape((x.shape[0],) + (-1, config['MINIBATCH_SIZE']) + x.shape[2:]), batch
            )
            sequences = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )
            permutation = jax.random.permutation(_rng, config['NUM_MINIBATCHES'])
            minibatches = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), sequences
            )
        else:
            permutation = jax.random.permutation(_rng, batch_size)
            batch = jax.tree.map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # jax.debug.print('aft2 {z}', z=jax.tree.map(lambda l: l.shape, shuffled_batch[0]), ordered=True)
            # shuffled_batch = batch
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [config['NUM_MINIBATCHES'], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
        # print(jax.tree.map(lambda x: x.shape, minibatches))
        # jax.debug.print('aft3 {z}', z=jax.tree.map(lambda l: l.shape, minibatches[0]), ordered=True)

        scanned_update_minibatch = nnx.scan(_update_minbatch,
                                            in_axes=((nnx.Carry, 0, 0, 0)))

        _, total_loss = scanned_update_minibatch((update_state.network, update_state.optimizer), *minibatches)

        # traj_batch, advantages, targets
        #
        # train_state, total_loss = jax.lax.scan(
        #     _update_minbatch, train_state, minibatches
        # )
        update_state = update_state._replace(rng=rng)
        # update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, total_loss

    runner_state.networks_policy.train()

    update_state = UpdateState(network=runner_state.networks_policy, optimizer=runner_state.optimizers_policy,
                               traj_batch=traj_batch, advantages=advantages, targets=targets, rng=runner_state.rng)

    scanned_update_epoch = nnx.scan(_update_epoch,
                                    in_axes=nnx.Carry,
                                    out_axes=(nnx.Carry, 0),
                                    length=config['UPDATE_EPOCHS'])

    update_state, loss_info = scanned_update_epoch(update_state)

    # update_state, loss_info = jax.lax.scan(
    #     _update_epoch, update_state, None, config['UPDATE_EPOCHS']
    # )

    # jax.debug.print('bat loss {x}', x=loss_info[0])

    runner_state.networks_policy.eval()

    runner_state = runner_state._replace(rng=update_state.rng)

    return runner_state, loss_info

# @partial(jax.jit, static_argnums=(0, 2, 3, 6))
def test_networks(env: RECEnv, train_state: TrainState, num_iter, config, rng, rec_rule_based_policy, curr_iter=0,
                  print_data=False):
    networks_policy, network_incentives = nnx.merge(train_state.graph_def, train_state.state)

    networks_policy.eval()
    network_incentives.eval()

    rng, _rng = jax.random.split(rng)

    obsv, env_state = env.reset(_rng, profile_index=0)

    if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic' and config['NUM_RL_AGENTS'] > 0:
        init_act_state_batteries, init_cri_state_batteries = networks_policy.get_initial_lstm_state()
        act_state_batteries, cri_state_batteries = init_act_state_batteries, init_cri_state_batteries
    else:
        act_state_batteries, cri_state_batteries = None, None

    # @scan_tqdm(num_iter, print_rate=num_iter // 100)
    def _env_step(runner_state, unused):
        obsv, env_state, act_state_batteries, cri_state_batteries, rng, next_profile_index = runner_state

        # print('aaaaa', obsv_batteries[:config['NUM_RL_AGENTS']].shape)

        actions = []

        obsv_rl = jax.tree.map(lambda x: x[:config['NUM_RL_AGENTS']], obsv)

        if config['NUM_RL_AGENTS'] > 0:

            if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
                pi, value, act_state_batteries, cri_state_batteries = networks_policy(obsv_rl, act_state_batteries,
                                                                                      cri_state_batteries)
            else:
                pi, value = networks_policy(obsv_rl)

            # deterministic action
            actions_rl = pi.mode()

            # print('act 1', actions_batteries_rl.shape)
            actions_rl = actions_rl.squeeze(axis=-1)
            actions.append(actions_rl)

        if config['NUM_BATTERY_FIRST_AGENTS'] > 0:
            idx_start_bf = config['NUM_RL_AGENTS']
            idx_end_bf = config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS']

            demand = obsv['demand'][idx_start_bf:idx_end_bf]
            generation = obsv['generation'][idx_start_bf:idx_end_bf]

            actions_battery_first = (generation - demand) / env_state.battery_states.electrical_state.v[
                                                            idx_start_bf:idx_end_bf]

            actions.append(actions_battery_first)

        if config['NUM_ONLY_MARKET_AGENTS'] > 0:
            actions_only_market = jnp.zeros(
                (config['NUM_ONLY_MARKET_AGENTS'],))
            actions.append(actions_only_market)

        if config['NUM_RANDOM_AGENTS'] > 0:
            rng, _rng = jax.random.split(rng)

            actions_random = jax.random.uniform(_rng,
                                                shape=(config['NUM_RANDOM_AGENTS'],),
                                                minval=-1.,
                                                maxval=1.)

            actions_random *= config['MAX_ACTION_RANDOM_AGENTS']

            actions.append(actions_random)

        actions = jnp.concat(actions, axis=0)

        actions_dict = {env.agents[i]: actions[i] for i in range(env.num_battery_agents)}

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward_env, done, info = env.step(
            _rng, env_state, actions_dict
        )

        info['actions'] = actions

        reward_env = jax.tree.map(lambda *vals: jnp.stack(vals, axis=0), *[reward_env[a] for a in env.agents])
        reward_env = {key: (val if key == 'r_glob' else jax.lax.stop_gradient(val)) for key, val in reward_env.items()}

        broadcast_actions = jnp.stack([actions]*config['NUM_RL_AGENTS'], axis=0)

        int_rewards_separate_frac = network_incentives(obsv_rl, broadcast_actions)
        print(int_rewards_separate_frac.shape)

        if config['NUM_RL_AGENTS'] < config['NUM_BATTERY_AGENTS']:
            n_ag = config['NUM_BATTERY_AGENTS']
            n_rl = config['NUM_RL_AGENTS']
            int_rewards_separate_frac_rule_based = jnp.eye(n_ag-n_rl, n_ag, n_rl)
            print(int_rewards_separate_frac_rule_based.shape)
            int_rewards_separate_frac =jnp.concat([int_rewards_separate_frac, int_rewards_separate_frac_rule_based], axis=0)

        print(reward_env)

        int_rewards_mat = int_rewards_separate_frac * reward_env['r_glob'][..., None]
        int_rewards = int_rewards_mat.sum(axis=-2)

        reward = jnp.stack([val for key, val in reward_env.items() if key != 'r_glob'], axis=0).sum(axis=0) + int_rewards

        info['int_rewards'] = int_rewards
        info['int_reward_mat'] = int_rewards_mat
        info['r_tot'] = reward

        info['dones'] = done

        rng, _rng = jax.random.split(rng)
        obsv, env_state, next_profile_index = jax.lax.cond(done['__all__'],
                                                           lambda: env.reset(_rng, profile_index=next_profile_index) + (
                                                               next_profile_index + 1,),
                                                           lambda: (obsv, env_state, next_profile_index))

        obsv = jax.tree.map(lambda *vals: jnp.stack(vals), *[obsv[a] for a in env.agents])

        runner_state = (obsv, env_state, act_state_batteries, cri_state_batteries, rng, next_profile_index)
        return runner_state, info

    obsv = jax.tree.map(lambda *vals: jnp.stack(vals), *[obsv[a] for a in env.agents])

    runner_state = (obsv, env_state, act_state_batteries, cri_state_batteries, rng, 1)

    runner_state, info = jax.lax.scan(_env_step, runner_state, jnp.arange(num_iter))

    reward_type = 'pure_reward'

    if print_data:
        jax.debug.print('curr_iter: {i}', i=curr_iter)
        for i in range(config['NUM_BATTERY_AGENTS']):
            jax.debug.print(
                '\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}\n\tr_glob: {r_glob}\n'
                '\tmean soc: {mean_soc}\n\tstd actions: {std_act}\n\tself consumption: {sc}\n',
                r_tot=jnp.sum(info['r_tot'][:, i]),
                r_trad=jnp.sum(info[reward_type]['r_trad'][:, i]),
                r_deg=jnp.sum(info[reward_type]['r_deg'][:, i]),
                r_clip=jnp.sum(info[reward_type]['r_clipping'][:, i]),
                r_glob=jnp.sum(info['int_rewards'][:, i]),
                mean_soc=jnp.mean(info['soc'][:, i]),
                std_act=jnp.std(info['actions'], axis=0),
                sc=jnp.sum(info['self_consumption']), ordered=True)

        jax.debug.print('glob reward matrix:\n{x}', x=info['int_reward_mat'].sum(axis=0), ordered=True)

        jax.debug.print('\n\tr_tot: {x}', x=jnp.sum(info['r_tot'][:, :config['NUM_RL_AGENTS']]), ordered=True)

    return info
