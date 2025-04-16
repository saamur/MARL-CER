from datetime import datetime

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

from jaxmarl.wrappers.baselines import JaxMARLWrapper

from algorithms.normalization_custom import RunningNorm
import algorithms.utils as utils
from ernestogym.envs_jax.multi_agent.env import RECEnv, EnvState


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


class VecEnvJaxMARL(JaxMARLWrapper):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0,))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0))

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


def make_train(config, env:RECEnv):
    config['NUM_UPDATES'] = (
        config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']
    )
    config['MINIBATCH_SIZE'] = (
        config['NUM_ENVS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES']
    )

    config['BATTERY_ACTION_SPACE_SIZE'] = env.action_space(env.battery_agents[0]).shape[0]
    config['BATTERY_OBSERVATION_SPACE_SIZE'] = env.observation_space(env.battery_agents[0]).shape[0]
    config['BATTERY_NUM_SEQUENCES'] = env.num_battery_obs_sequences
    config['BATTERY_OBS_IS_NORMALIZABLE'] = tuple(env.obs_is_normalizable_battery)

    config['REC_ACTION_SPACE_SIZE'] = env.action_space(env.rec_agent).shape[0]
    config['REC_OBS_KEYS'] = tuple(env.obs_rec_keys)
    config['NUM_BATTERY_AGENTS'] = env.num_battery_agents
    config['PASSIVE_HOUSES'] = (env.num_passive_houses>0)
    config['REC_OBS_IS_SEQUENCE'] = env.obs_is_sequence_rec
    config['REC_OBS_IS_LOCAL'] = env.obs_is_local_rec
    config['REC_OBS_IS_NORMALIZABLE'] = env.obs_is_normalizable_rec

    env = VecEnvJaxMARL(env)

    # env = LogWrapper(env)
    # env = ClipAction(env, low=env_params.i_min_action, high=env_params.i_max_action)
    # if config['NORMALIZE_ENV']:
    #     env = NormalizeVecObservation(env)
    #     env = NormalizeVecReward(env, config['GAMMA'])

    def schedule_builder(lr_init, lr_end, frac):

        tot_steps = int(config['NUM_MINIBATCHES'] * config['UPDATE_EPOCHS'] * config['NUM_UPDATES'] * frac)

        if config['LR_SCHEDULE'] == 'linear':
            return optax.schedules.linear_schedule(lr_init, lr_end, tot_steps)
        elif config['LR_SCHEDULE'] == 'cosine':
            return optax.schedules.cosine_decay_schedule(lr_init, tot_steps, lr_end/lr_init)
        else:
            return lr_init

    _rng = nnx.Rngs(123)
    network_batteries = utils.construct_battery_net_from_config_multi_agent(config, _rng)
    _rng = nnx.Rngs(222)
    network_rec = utils.construct_rec_net_from_config_multi_agent(config, _rng)

    schedule_batteries = schedule_builder(config['LR_BATTERIES'], config['LR_BATTERIES_MIN'], config['FRACTION_DYNAMIC_LR_BATTERIES'])
    schedule_rec = schedule_builder(config['LR_REC'], config['LR_REC_MIN'], config['FRACTION_DYNAMIC_LR_REC'])

    if config['USE_WEIGHT_DECAY']:
        tx_bat = optax.chain(
            optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
            optax.adamw(learning_rate=schedule_batteries, eps=1e-5),
        )
        tx_rec = optax.chain(
            optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
            optax.adamw(learning_rate=schedule_rec, eps=1e-5),
        )
    else:
        tx_bat = optax.chain(
            optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
            optax.adam(learning_rate=schedule_batteries, eps=1e-5),
        )
        tx_rec = optax.chain(
            optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
            optax.adam(learning_rate=schedule_rec, eps=1e-5),
        )


    optimizer_batteries = StackedOptimizer(config['NUM_BATTERY_AGENTS'], network_batteries, tx_bat)
    optimizer_rec = nnx.Optimizer(network_rec, tx_rec)

    graph_def, state = nnx.split((network_batteries, optimizer_batteries, network_rec, optimizer_rec))

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
    train_state: TrainState
    env_state: EnvState
    last_obs_batteries: jnp.ndarray
    rng: jax.random.PRNGKey
    done_prev_batteries: jnp.ndarray = jnp.array(False)
    done_prev_rec: jnp.ndarray = jnp.array(False)
    last_lstm_state_batteries: LSTMState = LSTMState()
    last_lstm_state_rec: LSTMState = LSTMState()


class Transition(NamedTuple):
    done_batteries: jnp.ndarray
    done_rec: jnp.ndarray

    actions_batteries: jnp.ndarray  #
    actions_rec: jnp.ndarray

    values_batteries: jnp.ndarray  #
    value_rec: jnp.ndarray

    rewards_batteries: jnp.ndarray
    reward_rec: jnp.ndarray

    log_prob_batteries: jnp.ndarray  #
    log_prob_rec: jnp.ndarray

    obs_batteries: jnp.ndarray  #
    obs_rec: jnp.ndarray

    info: jnp.ndarray

    done_prev_batteries: jnp.ndarray = 0
    done_prev_rec: jnp.ndarray = 0

    lstm_states_prev_batteries: LSTMState = LSTMState((), ())
    lstm_states_prev_rec: LSTMState = LSTMState((), ())


def train_wrapper(env:RECEnv, config, train_state:TrainState, rng, validate=True, freq_val=None, val_env=None, val_rng=None, val_num_iters=None, params=None, path_saving=None):

    infos = {}
    val_infos = {}

    dir_name = (datetime.now().strftime('%Y%m%d_%H%M%S') +
                '_bat_net_type_' + str(config['NETWORK_TYPE_BATTERIES']) +
                '_rec_net_type_' + str(config['NETWORK_TYPE_REC']) +
                '_lr_bat_' + str(config.get('LR_BATTERIES')) +
                '_lr_REC_' + str(config.get('LR_SCHEDULE')) +
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

        networks_batteries, _, network_rec, _ = nnx.merge(train_state.graph_def, train_state.state)

        val_info = jax.device_put(val_info, device=jax.devices('cpu')[0])
        jax.tree.map(update, val_infos, val_info)

        utils.save_state_multiagent(directory, networks_batteries, network_rec, config, is_checkpoint=True, num_steps=i)

    @partial(jax.jit, static_argnums=(0, 1, 4, 5, 6, 8))
    def train(env: RECEnv, config, train_state: TrainState, rng, validate=True, freq_val=None, val_env=None,
              val_rng=None, val_num_iters=None):
        # @partial(jax.jit, static_argnums=(0, 1, 4, 5, 6, 9), donate_argnums=(3,))
        # def train(env: RECEnv, config, train_state, rng, validate=True, freq_val=None, val_env=None, val_params=None,
        #               val_rng=None, val_num_iters=None):

        if validate:
            if freq_val is None or val_env is None or val_rng is None or val_num_iters is None:
                raise ValueError(
                    "'freq_val', 'val_env', 'val_rng' and 'val_num_iters' must be defined when 'validate' is True")

        rng, _rng = jax.random.split(rng)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config['NUM_ENVS'])
        obsv, env_state = env.reset(reset_rng)

        # jax.debug.print('beg', ordered=True)

        networks_batteries, _, network_rec, _ = nnx.merge(train_state.graph_def, train_state.state)

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            act_state_batteries, cri_state_batteries = networks_batteries.get_initial_lstm_state()
            act_state_batteries, cri_state_batteries = jax.tree.map(
                lambda x: jnp.tile(x[None, :], (config['NUM_ENVS'],) + (1,) * len(x.shape)),
                (act_state_batteries, cri_state_batteries))
            lstm_state_batteries = LSTMState(act_state_batteries, cri_state_batteries)

            episode_starts_batteries = jnp.ones((config['NUM_ENVS'], env.num_battery_agents), dtype=bool)
        else:
            lstm_state_batteries = LSTMState((jnp.ones(config['NUM_ENVS'], dtype=bool),),
                                             (jnp.ones(config['NUM_ENVS'], dtype=bool),))  # dummy
            episode_starts_batteries = jnp.ones(config['NUM_ENVS'], dtype=bool)  # dummy

        if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
            act_state_rec, cri_state_rec = network_rec.get_initial_lstm_state()
            act_state_rec, cri_state_rec = jax.tree.map(
                lambda x: jnp.tile(x[None, :], (config['NUM_ENVS'],) + (1,) * len(x.shape)),
                (act_state_rec, cri_state_rec))
            lstm_state_rec = LSTMState(act_state_rec, cri_state_rec)

            episode_starts_rec = jnp.ones((config['NUM_ENVS'],), dtype=bool)
        else:
            lstm_state_rec = LSTMState((jnp.ones(config['NUM_ENVS'], dtype=bool),),
                                       (jnp.ones(config['NUM_ENVS'], dtype=bool),))  # dummy
            episode_starts_rec = jnp.ones(config['NUM_ENVS'], dtype=bool)  # dummy


        # TRAIN LOOP
        @scan_tqdm(config['NUM_UPDATES'], print_rate=1)
        def _update_step(runner_state, curr_iter):
            # COLLECT TRAJECTORIES

            # jax.debug.print('\n\n\n{x}\n\n\n', x=curr_iter)

            runner_state, traj_batch, last_val_batteries, last_val_rec = collect_trajectories(runner_state, config, env)

            # jax.debug.print('traj taken', ordered=True)

            # jax.debug.print('{t}', t=jax.tree.map(lambda val: val.shape, traj_batch), ordered=True)

            # CALCULATE ADVANTAGE

            # last_val_batteries = traj_batch.values_batteries[-1]
            # last_val_rec = traj_batch.value_rec[-1]

            advantages, targets = _calculate_gae(traj_batch, last_val_batteries, last_val_rec, config)

            advantages_batteries, advantages_rec = advantages
            targets_batteries, targets_rec = targets

            # UPDATE NETWORKS
            rng_upd = runner_state.rng
            runner_state, total_loss_batteries = update_batteries_network(runner_state, traj_batch,
                                                                          advantages_batteries, targets_batteries,
                                                                          config)
            runner_state = runner_state._replace(rng=rng_upd)
            runner_state, total_loss_rec = update_rec_network(runner_state, traj_batch, advantages_rec, targets_rec,
                                                              config)

            metric = traj_batch.info

            if config.get('DEBUG'):

                def callback(info):
                    return_values = info['returned_episode_returns'][
                        info['returned_episode']
                    ]
                    timesteps = (
                            info['timestep'][info['returned_episode']] * config['NUM_ENVS']
                    )
                    for t in range(len(timesteps)):
                        print(
                            f'global step={timesteps[t]}, episodic return={return_values[t]}'
                        )

                jax.debug.callback(callback, metric)

            if validate:
                _ = jax.lax.cond(curr_iter % freq_val == 0,
                                 lambda: io_callback(update_val_info,
                                                     None,
                                                     test_networks(val_env, runner_state.train_state,
                                                                   val_num_iters, config, val_rng,
                                                                   curr_iter=curr_iter, print_data=True),
                                                     runner_state.train_state,
                                                     curr_iter // freq_val,
                                                     ordered=True),
                                 lambda: None)

            io_callback(end_update_step, None, metric, curr_iter, ordered=True)

            return runner_state, None

        rng, _rng = jax.random.split(rng)
        obsv_batteries = jnp.stack([obsv[a] for a in env.battery_agents], axis=1)

        runner_state = RunnerState(train_state=train_state,
                                   env_state=env_state,
                                   last_obs_batteries=obsv_batteries,
                                   rng=_rng,
                                   done_prev_batteries=episode_starts_batteries,
                                   done_prev_rec=episode_starts_rec,
                                   last_lstm_state_batteries=lstm_state_batteries,
                                   last_lstm_state_rec=lstm_state_rec)

        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config['NUM_UPDATES'])
        )

        networks_batteries, optimizer_batteries, network_rec, optimizer_rec = nnx.merge(
            runner_state.train_state.graph_def,
            runner_state.train_state.state)

        networks_batteries.eval()
        network_rec.eval()

        train_state = TrainState(graph_def=runner_state.train_state.graph_def,
                                 state=nnx.state((networks_batteries, optimizer_batteries, network_rec, optimizer_rec)))

        runner_state._replace(train_state=train_state)

        return runner_state

    runner_state = train(env, config, train_state, rng, validate, freq_val, val_env, val_rng, val_num_iters)

    print('Saving...')

    networks_batteries, _, network_rec, _ = nnx.merge(runner_state.train_state.graph_def, runner_state.train_state.state)

    utils.save_state_multiagent(directory, networks_batteries, network_rec, config, params, infos, val_infos, is_checkpoint=False)

    # metric = jax.tree.map(lambda *vals: np.stack(vals, axis=0), *infos)

    if validate:
        # val_info = jax.tree.map(lambda *vals: np.stack(vals, axis=0), *val_infos)
        return {'runner_state': runner_state, 'metrics': infos, 'val_info': val_infos}
    else:
        return {'runner_state': runner_state, 'metrics': infos}

def collect_trajectories(runner_state: RunnerState, config, env):
    networks_batteries, optimizer_batteries, network_rec, optimizer_rec = nnx.merge(runner_state.train_state.graph_def,
                                                                                    runner_state.train_state.state)

    networks_batteries.eval()
    network_rec.eval()

    def _env_step(carry):

        networks_batteries, network_rec, runner_state = carry

        # print(type(networks_batteries), type(optimizer_batteries), type(network_rec), type(optimizer_rec))

        # SELECT ACTION
        rng, _rng = jax.random.split(runner_state.rng)

        # _rng = jax.random.split(_rng, num=env.num_battery_agents)

        # print(f'aaa {config['BATTERY_OBSERVATION_SPACE_SIZE']}')
        # print(last_obs_batteries.shape)

        last_obs_batteries_num_batteries_first = jnp.swapaxes(runner_state.last_obs_batteries, 0, 1)
        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: jnp.where(runner_state.done_prev_batteries[(slice(None), slice(None)) + (None,)*(x.ndim-1)], x[None, :], y),
                                                          networks_batteries.get_initial_lstm_state(),
                                                          (runner_state.last_lstm_state_batteries.act_state, runner_state.last_lstm_state_batteries.cri_state))
            prev_act_state_num_batteries_first, prev_cri_state_num_batteries_first = jax.tree.map(lambda x : jnp.swapaxes(x, 0, 1), (prev_act_state, prev_cri_state))
            pi, value_batteries, lstm_act_state, lstm_cri_state = networks_batteries(last_obs_batteries_num_batteries_first, prev_act_state_num_batteries_first, prev_cri_state_num_batteries_first)
            lstm_act_state_batteries, lstm_cri_state_batteries = jax.tree.map(lambda x : jnp.swapaxes(x, 0, 1), (lstm_act_state, lstm_cri_state))
        else:
            pi, value_batteries = networks_batteries(last_obs_batteries_num_batteries_first)
            lstm_act_state_batteries, lstm_cri_state_batteries = runner_state.last_lstm_state_batteries.act_state, runner_state.last_lstm_state_batteries.cri_state

        actions_batteries = pi.sample(seed=_rng)                        # batteries first
        log_prob_batteries = pi.log_prob(actions_batteries)             # batteries first

        value_batteries, actions_batteries, log_prob_batteries = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1),
                                                                              (value_batteries, actions_batteries,
                                                                               log_prob_batteries))  # num_envs first

        actions_first = {env.battery_agents[i]: actions_batteries[:, i] for i in
                         range(env.num_battery_agents)}  # zip(env.battery_agents, actions_batteries)}
        actions_first[env.rec_agent] = jnp.zeros((config['NUM_ENVS'], env.num_battery_agents))

        # print(actions_first)

        # print(f'STATE {jax.tree.map(lambda x: x.shape, env_state)}')
        # print(f'ACTIONS {jax.tree.map(lambda x: x.shape, actions_first)}')

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config['NUM_ENVS'])
        env_state = runner_state.env_state
        obsv, env_state, reward_first, done_first, info_first = env.step(
            rng_step, env_state, actions_first
        )

        info_first['actions'] = actions_first

        rec_obsv = obsv[env.rec_agent]
        rng, _rng = jax.random.split(rng)

        # print(f'rec_obs {jax.tree.map(lambda x: x.shape, rec_obsv)}')

        if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
            prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: jnp.where(runner_state.done_prev_rec[(slice(None),) + (None,)*x.ndim], x[None, :], y),
                                                          network_rec.get_initial_lstm_state(),
                                                          (runner_state.last_lstm_state_rec.act_state, runner_state.last_lstm_state_rec.cri_state))
            pi, value_rec, lstm_act_state_rec, lstm_cri_state_rec = network_rec(rec_obsv, prev_act_state, prev_cri_state)
        else:
            pi, value_rec = network_rec(rec_obsv)
            lstm_act_state_rec, lstm_cri_state_rec = runner_state.last_lstm_state_rec.act_state, runner_state.last_lstm_state_batteries.cri_state



        actions_rec = pi.sample(seed=_rng)
        log_probs_rec = pi.log_prob(actions_rec + 1e-8)

        # jax.debug.print('alpha {t}', t=pi.concentration, ordered=True)
        # jax.debug.print('val {t}', t=value_rec, ordered=True)
        #
        # jax.debug.print('act {t}', t=actions_rec, ordered=True)

        # actions_rec = jnp.zeros((config['NUM_ENVS'], env.num_battery_agents))
        # log_probs_rec = jnp.zeros((config['NUM_ENVS'],))
        # value_rec = jnp.zeros((config['NUM_ENVS'],))

        actions_second = {agent: jnp.zeros((config['NUM_ENVS'], 1)) for agent in env.battery_agents}
        # actions_second[env.rec_agent] = actions_rec
        actions_second[env.rec_agent] = actions_rec

        assert actions_rec.shape == (config['NUM_ENVS'], env.num_battery_agents)

        # print(f'STATE {jax.tree.map(lambda x: x.shape, env_state)}')
        # print(f'ACTIONS {jax.tree.map(lambda x: x.shape, actions_second)}')

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config['NUM_ENVS'])
        obsv, env_state, reward_second, done_second, info_second = env.step(
            rng_step, env_state, actions_second
        )

        info_second['actions'] = actions_second

        done = jax.tree.map(jnp.logical_or, done_first, done_second)
        done_batteries = jnp.stack([done[a] for a in env.battery_agents], axis=1)
        done_rec = done[env.rec_agent]

        rewards_tot = jax.tree_map(lambda x, y: x + y, reward_first, reward_second)
        rewards_batteries = jnp.stack([rewards_tot[a] for a in env.battery_agents], axis=1)
        reward_rec = rewards_tot[env.rec_agent]

        info = jax.tree.map(lambda x, y: x + y, info_first, info_second)

        obs_batteries = jnp.stack([obsv[a] for a in env.battery_agents], axis=1)

        transition = Transition(
            done_batteries, done_rec,
            actions_batteries, actions_rec,
            value_batteries, value_rec,
            rewards_batteries, reward_rec,
            log_prob_batteries, log_probs_rec,
            runner_state.last_obs_batteries, rec_obsv,
            info,
            runner_state.done_prev_batteries, runner_state.done_prev_rec,
            runner_state.last_lstm_state_batteries, runner_state.last_lstm_state_rec
        )

        new_done_prev_batteries = done_batteries if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic' else runner_state.done_prev_batteries
        new_done_prev_rec = done_rec if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic' else runner_state.done_prev_rec


        # jax.debug.print('{t}', t=transition, ordered=True)
        runner_state = runner_state._replace(env_state=env_state,
                                             last_obs_batteries=obs_batteries,
                                             rng=rng,
                                             done_prev_batteries=new_done_prev_batteries,
                                             done_prev_rec=new_done_prev_rec,
                                             last_lstm_state_batteries=LSTMState(lstm_act_state_batteries,
                                                                                 lstm_cri_state_batteries),
                                             last_lstm_state_rec=LSTMState(lstm_act_state_rec,
                                                                           lstm_cri_state_rec))

        return (networks_batteries, network_rec, runner_state), transition

    # jax.debug.print('collecting traj', ordered=True)

    # runner_state, traj_batch = jax.lax.scan(
    #     _env_step, runner_state, None, config['NUM_STEPS']
    # )

    carry, traj_batch = nnx.scan(_env_step,
                                 in_axes=nnx.Carry,
                                 out_axes=(nnx.Carry, 0),
                                 length=config['NUM_STEPS'])((networks_batteries, network_rec, runner_state))

    networks_batteries, network_rec, runner_state = carry

    train_state = runner_state.train_state.replace(state=nnx.state((networks_batteries, optimizer_batteries, network_rec, optimizer_rec)))
    runner_state = runner_state._replace(train_state=train_state)

    _, transition = _env_step((networks_batteries, network_rec, runner_state))

    last_val_batteries = transition.values_batteries
    last_val_rec = transition.value_rec

    return runner_state, traj_batch, last_val_batteries, last_val_rec


def _calculate_gae(traj_batch, last_val_batteries, last_val_rec, config):

    def _get_advantages(gae_and_next_value, transition_data):
        gae, next_value = gae_and_next_value
        done, value, rewards = transition_data

        # delta = rewards + config['GAMMA'] * next_value * (1 - done) - value
        # gae = (delta + config['GAMMA'] * config['GAE_LAMBDA'] * (1 - done) * gae)

        delta = rewards + config['GAMMA'] * next_value - value
        gae = (delta + config['GAMMA'] * config['GAE_LAMBDA'] * gae)

        return (gae, value), gae

    rewards_batteries = traj_batch.rewards_batteries
    reward_rec = traj_batch.reward_rec

    assert rewards_batteries.shape[1] == config['NUM_ENVS']
    assert rewards_batteries.shape[2] == config['NUM_BATTERY_AGENTS']

    if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS']:
        rewards_batteries = (rewards_batteries - rewards_batteries.mean(axis=(0, 1), keepdims=True)) / (rewards_batteries.std(axis=(0, 1), keepdims=True) + 1e-8)
        reward_rec = (reward_rec - reward_rec.mean()) / (reward_rec.std() + 1e-8)

    _, advantages_batteries = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val_batteries), last_val_batteries),
        (traj_batch.done_batteries, traj_batch.values_batteries, rewards_batteries),
        reverse=True,
        unroll=32,
    )

    _, advantages_rec = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val_rec), last_val_rec),
        (traj_batch.done_rec, traj_batch.value_rec, reward_rec),
        reverse=True,
        unroll=32,
    )

    return ((advantages_batteries, advantages_rec),
            (advantages_batteries + traj_batch.values_batteries, advantages_rec + traj_batch.value_rec))

def update_batteries_network(runner_state, traj_batch, advantages, targets, config):
    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info

            # print(jax.tree.map(lambda x: x.shape, batch_info))

            def _loss_fn_batteries(network, traj_batch, gae, targets):
                traj_batch_data = (traj_batch.obs_batteries,
                                   traj_batch.actions_batteries,
                                   traj_batch.values_batteries,
                                   traj_batch.log_prob_batteries)
                traj_batch_obs, traj_batch_actions, traj_batch_values, traj_batch_log_probs = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_batch_data)

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

                assert value_loss.shape == (config['NUM_BATTERY_AGENTS'],)

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
                entropy = pi.entropy()

                entropy = entropy.mean(axis=1)

                total_loss = (
                        loss_actor
                        + config['VF_COEF'] * value_loss
                        - config['ENT_COEF'] * entropy
                )

                assert total_loss.ndim == 1
                # jax.debug.print('bat_loss', ordered=True)

                total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
                return total_loss, (value_loss, loss_actor, entropy)

            def _loss_fn_batteries_recurrent(network, traj_batch:Transition, gae, targets):

                normalized_obs_batteries_first = network.normalize_input(traj_batch.obs_batteries.swapaxes(0, 1))

                def forward_pass_lstm(carry, obs, beginning):
                    network, act_state, cri_state = carry

                    act_state, cri_state = jax.tree.map(lambda x, y: jnp.where(beginning[(slice(None),) + (None,)*(x.ndim-1)], x, y),
                                                        network.get_initial_lstm_state(),
                                                        (act_state, cri_state))
                    #jnp.where(beginning, init_states, states)
                    act_state, act_output = network.apply_lstm_act(obs, act_state)
                    cri_state, cri_output = network.apply_lstm_cri(obs, cri_state)
                    return (network, act_state, cri_state), act_output, cri_output

                _, act_outputs, cri_outputs = nnx.scan(forward_pass_lstm,
                                                       in_axes=(nnx.Carry, 1, 0),
                                                       out_axes=(nnx.Carry, 1, 1),
                                                       unroll=16)((network,) + jax.tree.map(lambda x: x[0], (traj_batch.lstm_states_prev_batteries.act_state, traj_batch.lstm_states_prev_batteries.cri_state)),
                                                                  normalized_obs_batteries_first, traj_batch.done_prev_batteries)

                pi = network.apply_act_mlp(normalized_obs_batteries_first, act_outputs)
                values = network.apply_cri_mlp(normalized_obs_batteries_first, cri_outputs)
                log_prob = pi.log_prob(traj_batch.actions_batteries.swapaxes(0, 1))

                values_time_first = values.swapaxes(0, 1)
                log_prob_time_first = log_prob.swapaxes(0, 1)

                if config['NORMALIZE_TARGETS']:
                    targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.values_batteries + (
                        values_time_first - traj_batch.values_batteries
                ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                value_losses = jnp.square(values_time_first - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                # jax.debug.print('value losses {x}', x=value_losses.shape)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=0)

                assert value_loss.shape == (config['NUM_BATTERY_AGENTS'],)

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob_time_first - traj_batch.log_prob_batteries)

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
                        - config['ENT_COEF'] * entropy
                )

                assert total_loss.ndim == 1

                total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
                return total_loss, (value_loss, loss_actor, entropy)

            networks_batteries, optimizer_batteries, network_rec, optimizer_rec = nnx.merge(train_state.graph_def,
                                                                                            train_state.state)

            networks_batteries.train()

            # traj_data_batteries_for_loss = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_data_batteries_for_loss)

            # print('traj_data_batteries_for_loss')
            # print(jax.tree_map(lambda x: x.shape, traj_data_batteries_for_loss))
            if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
                grad_fn_batteries = nnx.value_and_grad(_loss_fn_batteries_recurrent, has_aux=True)
                # traj_data_batteries_for_loss += (traj_batch.done_prev_batteries,)
            else:
                grad_fn_batteries = nnx.value_and_grad(_loss_fn_batteries, has_aux=True)

            total_loss_batteries, grads_batteries = grad_fn_batteries(
                networks_batteries,
                traj_batch,
                advantages,
                targets
            )
            # jax.debug.print('gggg {x}', x=grads_batteries.log_std)
            # print('tot loss batteries')
            # print(jax.tree_map(lambda x: x.shape, total_loss_batteries))
            # jax.debug.print('bat {x}', x=total_loss_batteries[0])
            #
            # jax.debug.print('{x}', x=optax.global_norm(grads_batteries), ordered=True)
            # jax.debug.print('{x}', x=total_loss_batteries[1], ordered=True)
            # jax.debug.print('{x}', x=networks_batteries.log_std.value, ordered=True)

            optimizer_batteries.update(grads_batteries)
            train_state = train_state.replace(
                state=nnx.state((networks_batteries, optimizer_batteries, network_rec, optimizer_rec)))

            total_loss = total_loss_batteries

            return train_state, total_loss

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config['MINIBATCH_SIZE'] * config['NUM_MINIBATCHES']
        assert (
                batch_size == config['NUM_STEPS'] * config['NUM_ENVS']
        ), 'batch size must be equal to number of steps * number of envs'

        batch = (traj_batch, advantages, targets)

        # print(jax.tree.map(lambda x: x.shape, batch))

        # jax.debug.print('bef {z}', z=jax.tree.map(lambda l: l.shape, traj_batch), ordered=True)

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), batch
            )
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((x.shape[0],) + (-1, config['MINIBATCH_SIZE']) + x.shape[2:]), batch
            )
            sequences = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )
            permutation = jax.random.permutation(_rng, config['NUM_MINIBATCHES'])
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), sequences
            )
        else:
            permutation = jax.random.permutation(_rng, batch_size)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # jax.debug.print('aft2 {z}', z=jax.tree.map(lambda l: l.shape, shuffled_batch[0]), ordered=True)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config['NUM_MINIBATCHES'], -1] + list(x.shape[1:])
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

    update_state = (runner_state.train_state, traj_batch, advantages, targets, runner_state.rng)

    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config['UPDATE_EPOCHS']
    )

    # jax.debug.print('bat loss {x}', x=loss_info[0])

    train_state = update_state[0]
    rng = update_state[-1]

    runner_state = runner_state._replace(train_state=train_state, rng=rng)

    return runner_state, loss_info


def update_rec_network(runner_state, traj_batch, advantages, targets, config):
    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info

            # print(jax.tree.map(lambda x: x.shape, batch_info))

            def _loss_fn_rec(network, traj_batch, gae, targets):

                traj_batch_obs = traj_batch.obs_rec
                traj_batch_actions = traj_batch.actions_rec
                traj_batch_values = traj_batch.value_rec
                traj_batch_log_probs = traj_batch.log_prob_rec

                # RERUN NETWORK
                pi, value = network(traj_batch_obs)
                log_prob = pi.log_prob(traj_batch_actions + 1e-8)
                # print(log_prob.shape)

                if config['NORMALIZE_TARGETS']:
                    targets = (targets - targets.mean()) / (targets.std() + 1e-8)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch_values + (
                        value - traj_batch_values
                ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch_log_probs)

                # jax.debug.print('rec ratio mean {x}, max {y}, min {z}, std {w}', x=ratio.mean(), y=ratio.max(),
                #                 z=ratio.min(), w=ratio.std(), ordered=True)

                if config['NORMALIZE_ADVANTAGES']:
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)

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
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = (
                        loss_actor
                        + config['VF_COEF'] * value_loss
                        - config['ENT_COEF'] * entropy
                )
                # jax.debug.print('rec_loss', ordered=True)
                return total_loss, (value_loss, loss_actor, entropy)

            def _loss_fn_rec_recurrent(network, traj_batch:Transition, gae, targets):
                # RERUN NETWORK

                data_for_network = network.prepare_data(traj_batch.obs_rec)

                def forward_pass_lstm(carry, network_input, beginning):
                    network, act_state, cri_state = carry

                    init_states = network.get_initial_lstm_state()

                    act_state, cri_state = jax.lax.cond(beginning, lambda: init_states, lambda: (act_state, cri_state))
                    act_state, act_output = network.apply_lstm_act(network_input, act_state)
                    cri_state, cri_output = network.apply_lstm_cri(network_input, cri_state)
                    return (network, act_state, cri_state), act_output, cri_output

                _, act_outputs, cri_outputs = nnx.scan(forward_pass_lstm,
                                                       in_axes=(nnx.Carry, 0, 0),
                                                       out_axes=(nnx.Carry, 0, 0),
                                                       unroll=16)((network,) + jax.tree.map(lambda x: x[0], (traj_batch.lstm_states_prev_rec.act_state, traj_batch.lstm_states_prev_rec.cri_state)),
                                                                  data_for_network, traj_batch.done_prev_rec)

                pi = network.apply_act_mlp(data_for_network, act_outputs)
                values = network.apply_cri_mlp(data_for_network, cri_outputs)

                log_prob = pi.log_prob(traj_batch.actions_rec + 1e-8)

                if config['NORMALIZE_TARGETS']:
                    targets = (targets - targets.mean()) / (targets.std() + 1e-8)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value_rec + (
                        values - traj_batch.value_rec
                ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                value_losses = jnp.square(values - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch.log_prob_rec)

                if config['NORMALIZE_ADVANTAGES']:
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                loss_actor1 = ratio * gae
                loss_actor2 = jnp.clip(ratio, 1.0 - config['CLIP_EPS'], 1.0 + config['CLIP_EPS']) * gae
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = loss_actor + config['VF_COEF'] * value_loss - config['ENT_COEF'] * entropy

                # jax.debug.print('{t}', t=total_loss, ordered=True)

                return total_loss, (value_loss, loss_actor, entropy)

            networks_batteries, optimizer_batteries, network_rec, optimizer_rec = nnx.merge(train_state.graph_def,
                                                                                            train_state.state)

            network_rec.train()

            # traj_data_batteries_for_loss = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_data_batteries_for_loss)

            # print('traj_data_batteries_for_loss')
            # print(jax.tree_map(lambda x: x.shape, traj_data_batteries_for_loss))
            if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
                grad_fn_rec = nnx.value_and_grad(_loss_fn_rec_recurrent, has_aux=True)
                # traj_data_batteries_for_loss += (traj_batch.done_prev_batteries,)
            else:
                grad_fn_rec = nnx.value_and_grad(_loss_fn_rec, has_aux=True)

            total_loss_rec, grads_rec = grad_fn_rec(
                network_rec,
                traj_batch,
                advantages,
                targets
            )

            # jax.debug.print('rec {x}', x=optax.global_norm(grads_rec), ordered=True)
            # jax.debug.print('rec {x}', x=total_loss_rec[1], ordered=True)

            # jax.lax.cond(jnp.isnan(total_loss_rec[0]).any(), lambda : jax.debug.breakpoint(), lambda : None)





            # def check_pytree(pytree):
            #     for key, leaf in jax.tree.leaves_with_path(pytree):
            #         checkify.check(jnp.logical_not(jnp.isfinite(leaf).all()), 'nope, the problem is' + str(key))
            #         jax.debug.print('{a}\t{x}', a=jnp.logical_not(jnp.isfinite(leaf).all()), x=key)
            #
            # jittable_check_pytree = checkify.checkify(check_pytree)
            #
            # jax.debug.breakpoint()
            #
            # jittable_check_pytree(nnx.state(network_rec, nnx.Param))
            # jittable_check_pytree(traj_batch)
            # jittable_check_pytree(advantages)
            # jittable_check_pytree(targets)
            # jittable_check_pytree(grads_rec)
            # jittable_check_pytree(total_loss_rec)

            # print('tot loss batteites')
            # print(jax.tree_map(lambda x: x.shape, total_loss_batteries))
            # jax.debug.print('rec {x}', x=total_loss_rec[0])

            optimizer_rec.update(grads_rec)
            train_state = train_state.replace(
                state=nnx.state((networks_batteries, optimizer_batteries, network_rec, optimizer_rec)))

            return train_state, total_loss_rec

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config['MINIBATCH_SIZE'] * config['NUM_MINIBATCHES']
        assert (
                batch_size == config['NUM_STEPS'] * config['NUM_ENVS']
        ), 'batch size must be equal to number of steps * number of envs'

        batch = (traj_batch, advantages, targets)

        # print(jax.tree.map(lambda x: x.shape, batch))

        # jax.debug.print('bef {z}', z=jax.tree.map(lambda l: l.shape, traj_batch), ordered=True)

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), batch
            )
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((x.shape[0],) + (-1, config['MINIBATCH_SIZE']) + x.shape[2:]), batch
            )
            sequences = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )
            permutation = jax.random.permutation(_rng, config['NUM_MINIBATCHES'])
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), sequences
            )
        else:
            permutation = jax.random.permutation(_rng, batch_size)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # jax.debug.print('aft2 {z}', z=jax.tree.map(lambda l: l.shape, shuffled_batch[0]), ordered=True)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config['NUM_MINIBATCHES'], -1] + list(x.shape[1:])
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

    update_state = (runner_state.train_state, traj_batch, advantages, targets, runner_state.rng)

    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config['UPDATE_EPOCHS']
    )

    # jax.debug.print('rec loss {x}', x=loss_info[0])

    train_state = update_state[0]
    rng = update_state[-1]

    runner_state = runner_state._replace(train_state=train_state, rng=rng)

    return runner_state, loss_info



# @partial(jax.jit, static_argnums=(0, 2, 3, 6))
def test_networks(env:RECEnv, train_state:TrainState, num_iter, config, rng, curr_iter=0, print_data=False):

    networks_batteries, _, network_rec, _ = nnx.merge(train_state.graph_def, train_state.state)

    networks_batteries.eval()
    network_rec.eval()

    rng, _rng = jax.random.split(rng)

    obsv, env_state = env.reset(_rng, profile_index=0)

    if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
        init_act_state_batteries, init_cri_state_batteries = networks_batteries.get_initial_lstm_state()
        act_state_batteries, cri_state_batteries = init_act_state_batteries, init_cri_state_batteries
    else:
        act_state_batteries, cri_state_batteries = None, None

    if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
        init_act_state_rec, init_cri_state_rec = network_rec.get_initial_lstm_state()
        act_state_rec, cri_state_rec = init_act_state_rec, init_cri_state_rec
    else:
        act_state_rec, cri_state_rec = None, None

    @scan_tqdm(num_iter, print_rate=num_iter // 100)
    def _env_step(runner_state, unused):
        obsv_batteries, env_state, act_state_batteries, act_state_rec, rng, next_profile_index = runner_state

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            pi, _, act_state_batteries, _ = networks_batteries(obsv_batteries, act_state_batteries, cri_state_batteries)
        else:
            pi, _ = networks_batteries(obsv_batteries)

        #deterministic action
        actions_batteries = pi.mode()
        actions_batteries = actions_batteries.squeeze(axis=-1)

        actions_first = {env.battery_agents[i]: actions_batteries[i] for i in range(env.num_battery_agents)}
        actions_first[env.rec_agent] = jnp.zeros(env.num_battery_agents)

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward_first, done_first, info_first = env.step(
            _rng, env_state, actions_first
        )

        rec_obsv = obsv[env.rec_agent]

        if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
            pi, _, act_state_rec, _ = network_rec(rec_obsv, act_state_rec, cri_state_rec)
        else:
            pi, _ = network_rec(rec_obsv)

        actions_rec = pi.mean()

        actions_second = {agent: jnp.array(0.) for agent in env.battery_agents}
        actions_second[env.rec_agent] = actions_rec

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward_second, done_second, info_second = env.step(
            _rng, env_state, actions_second
        )

        done = jnp.logical_or(done_first['__all__'], done_second['__all__'])

        info = jax.tree.map(lambda  x, y: x + y, info_first, info_second)

        info['actions_batteries'] = actions_batteries
        info['actions_rec'] = actions_rec
        info['dones'] = jax.tree.map(lambda x, y : jnp.logical_or(x, y), done_first, done_second)

        rng, _rng = jax.random.split(rng)
        obsv, env_state,next_profile_index = jax.lax.cond(done,
                                                          lambda : env.reset(_rng, profile_index=next_profile_index) + (next_profile_index+1,),
                                                          lambda : (obsv, env_state, next_profile_index))

        # jax.lax.cond(done, lambda: jax.debug.print('i {x}, {dem}, {gen}, {spr}, {bpr}, {rew}, {pr}, {nr}, {wr}\n{soh}\n',
        #                                            x=unused, dem=info['demands'], gen=info['generations'],
        #                                            spr=info['sell_prices'], bpr=info['buy_prices'], rew=info['r_tot'],
        #                                            pr=info['pure_reward'], nr=info['norm_reward'], wr=info['weig_reward'],
        #                                            soh=info['soh'], ordered=True), lambda : None)

        obs_batteries = jnp.vstack([obsv[a] for a in env.battery_agents])

        runner_state = (obs_batteries, env_state, act_state_batteries, act_state_rec, rng, next_profile_index)
        return runner_state, info

    obsv_batteries = jnp.vstack([obsv[a] for a in env.battery_agents])

    runner_state = (obsv_batteries, env_state, act_state_batteries, act_state_rec, rng, 1)

    runner_state, info = jax.lax.scan(_env_step, runner_state, jnp.arange(num_iter))

    reward_type = 'weig_reward'

    if print_data:
        jax.debug.print('curr_iter: {i}\n\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}\n\tr_glob: {r_glob}\n\tr_rec: {r_rec}\n'
                        '\tmean soc: {mean_soc}\n\tstd actions: {std_act}\n\tself consumption: {sc}\n',
                        i=curr_iter, r_tot=jnp.sum(info['r_tot']), r_trad=jnp.sum(info[reward_type]['r_trad']),
                        r_deg=jnp.sum(info[reward_type]['r_deg']), r_clip=jnp.sum(info[reward_type]['r_clipping']),
                        r_glob=jnp.sum(info['r_glob']), r_rec=jnp.sum(info['rec_reward']),
                        mean_soc=jnp.mean(info['soc']), std_act=jnp.std(info['actions_batteries'], axis=0),
                        sc=jnp.sum(info['self_consumption']))

    return info