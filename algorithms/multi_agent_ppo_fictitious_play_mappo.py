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

from jaxmarl.wrappers.baselines import JaxMARLWrapper

import algorithms.utils as utils
from ernestogym.envs_jax.multi_agent.env import RECEnv, EnvState
from algorithms.networks import StackedActorCritic, StackedRecurrentActorCritic, RECActorCritic, RECRecurrentActorCritic


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


def make_train(config, env:RECEnv, network_batteries=None):

    print('PPO FICTITIOUS')

    if 'NUM_MINIBATCHES_BATTERIES' not in config.keys():
        if 'NUM_MINIBATCHES' not in config.keys():
            raise ValueError("At least one of config['NUM_MINIBATCHES_BATTERIES'] and config['NUM_MINIBATCHES'] must be provided")
        config['NUM_MINIBATCHES_BATTERIES'] = config['NUM_MINIBATCHES']

    if 'NUM_MINIBATCHES_REC' not in config.keys():
        if 'NUM_MINIBATCHES' not in config.keys():
            raise ValueError("At least one of config['NUM_MINIBATCHES_REC'] and config['NUM_MINIBATCHES'] must be provided")
        config['NUM_MINIBATCHES_REC'] = config['NUM_MINIBATCHES']

    config['NUM_UPDATES'] = config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']
    config['MINIBATCH_SIZE_BATTERIES'] = config['NUM_ENVS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES_BATTERIES']
    config['MINIBATCH_SIZE_REC'] = config['NUM_ENVS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES_REC']

    config['BATTERY_ACTION_SPACE_SIZE'] = env.action_space(env.battery_agents[0]).shape[0]
    config['BATTERY_OBS_KEYS'] = tuple(env.obs_battery_agents_keys)
    config['BATTERY_OBS_IS_SEQUENCE'] = env.obs_is_sequence_battery
    config['BATTERY_OBS_IS_NORMALIZABLE'] = env.obs_is_normalizable_battery

    if config.get('REC_VALUE_IN_BATTERY_OBS', False) or config.get('REC_VALUE_IN_BATTERY_OBS_CRI', False):
        if config.get('REC_VALUE_IN_BATTERY_OBS', False):
            config['BATTERY_OBS_KEYS'] += ('rec_value',)
        else:
            config['BATTERY_OBS_KEYS_CRI'] = config['BATTERY_OBS_KEYS'] + ('rec_value',)
        config['BATTERY_OBS_IS_SEQUENCE']['rec_value'] = True
        config['BATTERY_OBS_IS_NORMALIZABLE']['rec_value'] = True


    if 'NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_BATTERIES' not in config.keys():
        if 'NORMALIZE_REWARD_FOR_GAE_AND_TARGETS' not in config.keys():
            raise ValueError("At least one of config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_BATTERIES'] and config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS'] must be provided")
        config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_BATTERIES'] = config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS']

    if 'NORMALIZE_TARGETS_BATTERIES' not in config.keys():
        if 'NORMALIZE_TARGETS' not in config.keys():
            raise ValueError("At least one of config['NORMALIZE_TARGETS_BATTERIES'] and config['NORMALIZE_TARGETS'] must be provided")
        config['NORMALIZE_TARGETS_BATTERIES'] = config['NORMALIZE_TARGETS']

    if 'NORMALIZE_ADVANTAGES_BATTERIES' not in config.keys():
        if 'NORMALIZE_ADVANTAGES' not in config.keys():
            raise ValueError("At least one of config['NORMALIZE_ADVANTAGES_BATTERIES'] and config['NORMALIZE_ADVANTAGES'] must be provided")
        config['NORMALIZE_ADVANTAGES_BATTERIES'] = config['NORMALIZE_ADVANTAGES']

    if 'ENT_COEF_BATTERIES' not in config.keys():
        if 'ENT_COEF' not in config.keys():
            raise ValueError("At least one of config['ENT_COEF_BATTERIES'] and config['ENT_COEF'] must be provided")
        config['ENT_COEF_BATTERIES'] = config['ENT_COEF']

    config['REC_ACTION_SPACE_SIZE'] = env.action_space(env.rec_agent).shape[0]
    config['REC_OBS_KEYS'] = tuple(env.obs_rec_keys)
    config['NUM_BATTERY_AGENTS'] = env.num_battery_agents
    config['PASSIVE_HOUSES'] = (env.num_passive_houses>0)
    config['REC_OBS_IS_SEQUENCE'] = env.obs_is_sequence_rec
    config['REC_OBS_IS_LOCAL'] = env.obs_is_local_rec
    config['REC_OBS_IS_NORMALIZABLE'] = env.obs_is_normalizable_rec

    if config.get('BATTERY_VALUES_IN_REC_OBS', False) or config.get('BATTERY_VALUES_IN_REC_OBS_CRI', False):
        if config.get('BATTERY_VALUES_IN_REC_OBS', False):
            config['REC_OBS_KEYS'] += ('battery_values',)
        else:
            config['REC_OBS_KEYS_CRI'] = config['REC_OBS_KEYS'] + ('battery_values',)
        config['REC_OBS_IS_SEQUENCE']['battery_values'] = True
        config['REC_OBS_IS_LOCAL']['battery_values'] = True
        config['REC_OBS_IS_NORMALIZABLE']['battery_values'] = True

    if 'NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_REC' not in config.keys():
        if 'NORMALIZE_REWARD_FOR_GAE_AND_TARGETS' not in config.keys():
            raise ValueError(
                "At least one of config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_REC'] and config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS'] must be provided")
        config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_REC'] = config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS']

    if 'NORMALIZE_TARGETS_REC' not in config.keys():
        if 'NORMALIZE_TARGETS' not in config.keys():
            raise ValueError(
                "At least one of config['NORMALIZE_TARGETS_REC'] and config['NORMALIZE_TARGETS'] must be provided")
        config['NORMALIZE_TARGETS_REC'] = config['NORMALIZE_TARGETS']

    if 'NORMALIZE_ADVANTAGES_REC' not in config.keys():
        if 'NORMALIZE_ADVANTAGES' not in config.keys():
            raise ValueError(
                "At least one of config['NORMALIZE_ADVANTAGES_REC'] and config['NORMALIZE_ADVANTAGES'] must be provided")
        config['NORMALIZE_ADVANTAGES_REC'] = config['NORMALIZE_ADVANTAGES']

    if 'ENT_COEF_REC' not in config.keys():
        if 'ENT_COEF' not in config.keys():
            raise ValueError("At least one of config['ENT_COEF_REC'] and config['ENT_COEF'] must be provided")
        config['ENT_COEF_REC'] = config['ENT_COEF']

    assert (len(env.battery_agents) ==
            config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS'] +
            config['NUM_ONLY_MARKET_AGENTS'] + config['NUM_RANDOM_AGENTS'])

    env = VecEnvJaxMARL(env)

    # env = LogWrapper(env)
    # env = ClipAction(env, low=env_params.i_min_action, high=env_params.i_max_action)
    # if config['NORMALIZE_ENV']:
    #     env = NormalizeVecObservation(env)
    #     env = NormalizeVecReward(env, config['GAMMA'])

    def schedule_builder(lr_init, lr_end, frac_dynamic, num_updates, num_minibatches, warm_up):

        tot_steps = int(num_minibatches * config['UPDATE_EPOCHS'] * num_updates * frac_dynamic)
        warm_up_steps = int(num_minibatches * config['UPDATE_EPOCHS'] * num_updates * warm_up)

        if config['LR_SCHEDULE'] == 'linear':
            return optax.schedules.linear_schedule(lr_init, lr_end, tot_steps)
        elif config['LR_SCHEDULE'] == 'cosine':
            optax.schedules.cosine_decay_schedule(lr_init, tot_steps, lr_end / lr_init)
            return optax.schedules.warmup_cosine_decay_schedule(0., lr_init, warm_up_steps, tot_steps, lr_end)
        else:
            return lr_init

    if network_batteries is None:
        _rng = nnx.Rngs(123)
        network_batteries = utils.construct_battery_net_from_config_multi_agent(config, _rng, num_nets=config['NUM_RL_AGENTS'])
    _rng = nnx.Rngs(222)
    network_rec = utils.construct_rec_net_from_config_multi_agent(config, _rng)

    schedule_batteries = schedule_builder(config['LR_BATTERIES'],
                                          config['LR_BATTERIES_MIN'],
                                          config['FRACTION_DYNAMIC_LR_BATTERIES'],
                                          config['NUM_UPDATES'] * config['NUM_CONSECUTIVE_UPDATES_BATTERIES']/(config['NUM_CONSECUTIVE_UPDATES_BATTERIES']+config['NUM_CONSECUTIVE_UPDATES_REC']),
                                          config['NUM_MINIBATCHES_BATTERIES'],
                                          warm_up=config.get('WARMUP_SCHEDULE_BATTERIES', 0))
    schedule_rec = schedule_builder(config['LR_REC'],
                                    config['LR_REC_MIN'],
                                    config['FRACTION_DYNAMIC_LR_REC'],
                                    config['NUM_UPDATES'] * config['NUM_CONSECUTIVE_UPDATES_REC']/(config['NUM_CONSECUTIVE_UPDATES_BATTERIES']+config['NUM_CONSECUTIVE_UPDATES_REC']),
                                    config['NUM_MINIBATCHES_REC'],
                                    warm_up=config.get('WARMUP_SCHEDULE_REC', 0))

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


    optimizer_batteries = StackedOptimizer(config['NUM_RL_AGENTS'], network_batteries, tx_bat)
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
    # train_state: TrainState
    network_batteries: Union[StackedActorCritic, StackedRecurrentActorCritic]
    network_rec: Union[RECActorCritic, RECRecurrentActorCritic]
    optimizer_batteries: StackedOptimizer
    optimizer_rec: nnx.Optimizer

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


def train_wrapper(env:RECEnv, config, network_batteries, optimizer_batteries, network_rec, optimizer_rec,
                  rng, world_metadata, rec_rule_based_policy=None, validate=True, freq_val=None, val_env=None, val_rng=None,
                  val_num_iters=None, path_saving=None):

    if config['USE_REC_RULE_BASED_POLICY'] and rec_rule_based_policy is None:
        raise ValueError("when config['USE_REC_RULE_BASED_POLICY'] is True, rec_rule_based_policy must not be None")

    if not config['USE_REC_RULE_BASED_POLICY'] and (network_rec is None or optimizer_rec is None):
        raise ValueError("when config['USE_REC_RULE_BASED_POLICY'] is False, network_rec and optimizer_rec must not be None")

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

        # network_batteries, _, network_rec, _ = nnx.merge(train_state.graph_def, train_state.state)
        network_batteries, network_rec = nnx.merge(train_state.graph_def, train_state.state)

        val_info = jax.device_put(val_info, device=jax.devices('cpu')[0])
        jax.tree.map(update, val_infos, val_info)

        utils.save_state_multiagent(directory, network_batteries, network_rec, config, world_metadata, is_checkpoint=True, num_steps=i)

    @partial(nnx.jit, static_argnums=(0, 1, 7, 8, 9, 11))
    def train(env: RECEnv, config, network_batteries, optimizer_batteries, network_rec, optimizer_rec, rng, validate=True, freq_val=None, val_env=None,
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

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic' and config['NUM_RL_AGENTS'] > 0:
            act_state_batteries, cri_state_batteries = network_batteries.get_initial_lstm_state()
            act_state_batteries, cri_state_batteries = jax.tree.map(
                lambda x: jnp.tile(x[None, :], (config['NUM_ENVS'],) + (1,) * len(x.shape)),
                (act_state_batteries, cri_state_batteries))
            lstm_state_batteries = LSTMState(act_state_batteries, cri_state_batteries)

            episode_starts_batteries = jnp.ones((config['NUM_ENVS'], config['NUM_RL_AGENTS']), dtype=bool)
        else:
            lstm_state_batteries = LSTMState((jnp.ones(config['NUM_ENVS'], dtype=bool),),
                                             (jnp.ones(config['NUM_ENVS'], dtype=bool),))  # dummy
            episode_starts_batteries = jnp.ones(config['NUM_ENVS'], dtype=bool)  # dummy

        if not config['USE_REC_RULE_BASED_POLICY'] and config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
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
        # @scan_tqdm(config['NUM_UPDATES'], print_rate=1)
        @tqdm_custom(0, 0, 1, config['NUM_UPDATES'], print_rate=1)
        def _update_step(runner_state, curr_iter):
            # COLLECT TRAJECTORIES

            # jax.debug.print('iter: {x}', x=curr_iter)

            def update_batteries(runner_state):
                runner_state, traj_batch, last_val_batteries, last_val_rec = collect_trajectories(runner_state, config, env, True)

                advantages, targets = _calculate_gae(traj_batch, last_val_batteries, last_val_rec, config)

                advantages_batteries, _ = advantages
                targets_batteries, _ = targets

                runner_state, total_loss_batteries = update_batteries_network(runner_state, traj_batch,
                                                                              advantages_batteries, targets_batteries,
                                                                              config)

                return runner_state, traj_batch.info

            def update_rec(runner_state):

                runner_state, traj_batch, last_val_batteries, last_val_rec = collect_trajectories(runner_state, config, env, False)

                advantages, targets = _calculate_gae(traj_batch, last_val_batteries, last_val_rec, config)

                _, advantages_rec = advantages
                _, targets_rec = targets

                runner_state, total_loss_rec = update_rec_network(runner_state, traj_batch, advantages_rec,
                                                                  targets_rec, config)

                return runner_state, traj_batch.info

            runner_state, info = nnx.cond(curr_iter % (config['NUM_CONSECUTIVE_UPDATES_BATTERIES']+config['NUM_CONSECUTIVE_UPDATES_REC']) < config['NUM_CONSECUTIVE_UPDATES_BATTERIES'],
                                          update_batteries,
                                          update_rec,
                                          runner_state)



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

                jax.debug.callback(callback, info)

            if validate:

                # print('boooooh', TrainState(*nnx.split((runner_state.network_batteries, runner_state.optimizer_batteries, runner_state.network_rec, runner_state.optimizer_rec))))
                jax.lax.cond(curr_iter % freq_val == 0,
                             lambda: io_callback(update_val_info,
                                                 None,
                                                 test_networks(val_env, TrainState(*nnx.split((runner_state.network_batteries, runner_state.optimizer_batteries, runner_state.network_rec, runner_state.optimizer_rec))),
                                                               val_num_iters, config, val_rng, rec_rule_based_policy=rec_rule_based_policy,
                                                               curr_iter=curr_iter, print_data=True),
                                                 # None,
                                                 # (runner_state.network_batteries, runner_state.optimizer_batteries, runner_state.network_rec, runner_state.optimizer_rec),
                                                 TrainState(*nnx.split((runner_state.network_batteries, runner_state.network_rec))),
                                                 # TrainState(*nnx.split((runner_state.network_batteries, runner_state.optimizer_batteries, runner_state.network_rec, runner_state.optimizer_rec))),
                                                 curr_iter // freq_val,
                                                 ordered=True),
                             lambda: None)

            if config.get('SAVE_TRAIN_INFO', False):
                io_callback(end_update_step, None, info, curr_iter, ordered=True)

            return runner_state

        rng, _rng = jax.random.split(rng)
        obsv_batteries = jax.tree.map(lambda *vals: jnp.stack(vals, axis=1), *[obsv[a] for a in env.battery_agents])

        if config.get('REC_VALUE_IN_BATTERY_OBS', False) or config.get('REC_VALUE_IN_BATTERY_OBS_CRI', False):
            obsv_batteries['rec_value'] = jnp.zeros((config['NUM_ENVS'], config['NUM_BATTERY_AGENTS']))

        if config['NUM_RL_AGENTS'] > 0:
            network_batteries.eval()
        if not config['USE_REC_RULE_BASED_POLICY']:
            network_rec.eval()

        runner_state = RunnerState(network_batteries=network_batteries,
                                   network_rec=network_rec,
                                   optimizer_batteries=optimizer_batteries,
                                   optimizer_rec=optimizer_rec,
                                   env_state=env_state,
                                   last_obs_batteries=obsv_batteries,
                                   rng=_rng,
                                   done_prev_batteries=episode_starts_batteries,
                                   done_prev_rec=episode_starts_rec,
                                   last_lstm_state_batteries=lstm_state_batteries,
                                   last_lstm_state_rec=lstm_state_rec)

        scanned_update_step = nnx.scan(_update_step,
                                       in_axes=(nnx.Carry, 0),
                                       out_axes=nnx.Carry)

        runner_state = scanned_update_step(runner_state, jnp.arange(config['NUM_UPDATES']))

        # runner_state, _ = jax.lax.scan(
        #     _update_step, runner_state, jnp.arange(config['NUM_UPDATES'])
        # )

        runner_state.network_batteries.eval()
        if not config['USE_REC_RULE_BASED_POLICY']:
            runner_state.network_rec.eval()

        return runner_state

    runner_state = train(env, config, network_batteries, optimizer_batteries, network_rec, optimizer_rec, rng, validate, freq_val, val_env, val_rng, val_num_iters)

    print('Saving...')

    t0 = time()

    utils.save_state_multiagent(directory, runner_state.network_batteries, runner_state.network_rec, config, world_metadata, infos, val_infos, is_checkpoint=False)

    # metric = jax.tree.map(lambda *vals: np.stack(vals, axis=0), *infos)

    print(f'Saving time: {t0-time():.2f} s')

    if validate:
        # val_info = jax.tree.map(lambda *vals: np.stack(vals, axis=0), *val_infos)
        return {'runner_state': runner_state, 'metrics': infos, 'val_info': val_infos}
    else:
        return {'runner_state': runner_state, 'metrics': infos}

def collect_trajectories(runner_state: RunnerState, config, env, for_batteries_update):

    def _env_step(runner_state: RunnerState):

        network_batteries, network_rec = runner_state.network_batteries, runner_state.network_rec

        # print(type(network_batteries), type(optimizer_batteries), type(network_rec), type(optimizer_rec))

        # SELECT ACTION
        rng, _rng = jax.random.split(runner_state.rng)

        # _rng = jax.random.split(_rng, num=env.num_battery_agents)

        # print(f'aaa {config['BATTERY_OBSERVATION_SPACE_SIZE']}')
        # print(last_obs_batteries.shape)

        actions_batteries = []

        last_obs_batteries_rl_num_batteries_first = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1)[:config['NUM_RL_AGENTS']], runner_state.last_obs_batteries)

        print(last_obs_batteries_rl_num_batteries_first)

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':

            prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: jnp.where(runner_state.done_prev_batteries[(slice(None), slice(None)) + (None,)*(x.ndim-1)], x[None, :], y),
                                                          network_batteries.get_initial_lstm_state(),
                                                          (runner_state.last_lstm_state_batteries.act_state, runner_state.last_lstm_state_batteries.cri_state))
            prev_act_state_num_batteries_first, prev_cri_state_num_batteries_first = jax.tree.map(lambda x : jnp.swapaxes(x, 0, 1), (prev_act_state, prev_cri_state))
            pi, value_batteries, lstm_act_state, lstm_cri_state = network_batteries(last_obs_batteries_rl_num_batteries_first, prev_act_state_num_batteries_first, prev_cri_state_num_batteries_first)
            lstm_act_state_batteries, lstm_cri_state_batteries = jax.tree.map(lambda x : jnp.swapaxes(x, 0, 1), (lstm_act_state, lstm_cri_state))
        else:
            pi, value_batteries = network_batteries(last_obs_batteries_rl_num_batteries_first, obs_for_cri=runner_state.last_obs_batteries)
            lstm_act_state_batteries, lstm_cri_state_batteries = runner_state.last_lstm_state_batteries.act_state, runner_state.last_lstm_state_batteries.cri_state

        actions_batteries_rl = pi.sample(seed=_rng) if for_batteries_update else pi.mean()                        # batteries first
        log_prob_batteries = pi.log_prob(actions_batteries_rl)             # batteries first

        value_batteries, actions_batteries_rl, log_prob_batteries = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1),
                                                                              (value_batteries, actions_batteries_rl,
                                                                               log_prob_batteries))  # num_envs first

        actions_batteries.append(actions_batteries_rl)

        if config['NUM_BATTERY_FIRST_AGENTS'] > 0:
            idx_start_bf = config['NUM_RL_AGENTS']
            idx_end_bf = config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS']

            demand = runner_state.last_obs_batteries['demand'][:, idx_start_bf:idx_end_bf]
            generation = runner_state.last_obs_batteries['generation'][:, idx_start_bf:idx_end_bf]
            actions_batteries_battery_first = (generation - demand) / runner_state.env_state.battery_states.electrical_state.v[:, idx_start_bf:idx_end_bf]

            actions_batteries_battery_first = jnp.expand_dims(actions_batteries_battery_first, -1)

            print(actions_batteries_battery_first.shape)

            actions_batteries.append(actions_batteries_battery_first)

        if config['NUM_ONLY_MARKET_AGENTS'] > 0:
            actions_batteries_only_market = jnp.zeros((config['NUM_ENVS'], config['NUM_ONLY_MARKET_AGENTS'], config['BATTERY_ACTION_SPACE_SIZE']))
            print(actions_batteries_only_market.shape)
            actions_batteries.append(actions_batteries_only_market)

        if config['NUM_RANDOM_AGENTS'] > 0:
            rng, _rng = jax.random.split(rng)

            actions_batteries_random = jax.random.uniform(_rng,
                                                          shape=(config['NUM_ENVS'], config['NUM_RANDOM_AGENTS']),
                                                          minval=-1., maxval=1.)

            actions_batteries_random *= config['MAX_ACTION_RANDOM_AGENTS']

            actions_batteries_random = jnp.expand_dims(actions_batteries_random, -1)

            actions_batteries.append(actions_batteries_random)

        actions_batteries = jnp.concat(actions_batteries, axis=1)

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

        if config.get('BATTERY_VALUES_IN_REC_OBS', False) or config.get('BATTERY_VALUES_IN_REC_OBS_CRI', False):

            if config['NUM_RL_AGENTS'] < config['NUM_BATTERY_AGENTS']:
                values_batteries_for_rec = jnp.concat((value_batteries, jnp.zeros((config['NUM_ENVS'], config['NUM_BATTERY_AGENTS'] - config['NUM_RL_AGENTS']))), axis=1)
            else:
                values_batteries_for_rec = value_batteries

            rec_obsv['battery_values'] = values_batteries_for_rec

        # jax.debug.print('rec obsv {x}', x=rec_obsv)

        # print(f'rec_obs {jax.tree.map(lambda x: x.shape, rec_obsv)}')

        if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
            prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: jnp.where(runner_state.done_prev_rec[(slice(None),) + (None,)*x.ndim], x[None, :], y),
                                                          network_rec.get_initial_lstm_state(),
                                                          (runner_state.last_lstm_state_rec.act_state, runner_state.last_lstm_state_rec.cri_state))
            pi, value_rec, lstm_act_state_rec, lstm_cri_state_rec, separate_cri = network_rec(rec_obsv, prev_act_state, prev_cri_state, return_separate_cri=True)
        else:
            pi, value_rec, separate_cri = network_rec(rec_obsv, return_separate_cri=True)
            lstm_act_state_rec, lstm_cri_state_rec = runner_state.last_lstm_state_rec.act_state, runner_state.last_lstm_state_rec.cri_state

        actions_rec = pi.sample(seed=_rng) if not for_batteries_update else pi.mean()
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

        rewards_tot = jax.tree.map(lambda x, y: x + y, reward_first, reward_second)
        rewards_batteries = jnp.stack([rewards_tot[a] for a in env.battery_agents], axis=1)
        reward_rec = rewards_tot[env.rec_agent]

        info = jax.tree.map(lambda x, y: x + y, info_first, info_second)

        obs_batteries = jax.tree.map(lambda *vals: jnp.stack(vals, axis=1), *[obsv[a] for a in env.battery_agents])

        if config.get('REC_VALUE_IN_BATTERY_OBS', False) or config.get('REC_VALUE_IN_BATTERY_OBS_CRI', False):
            obs_batteries['rec_value'] = separate_cri

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

        new_done_prev_batteries = done_batteries[:, :config['NUM_RL_AGENTS']] if (config['NUM_RL_AGENTS']>0 and config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic') else runner_state.done_prev_batteries
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

        return runner_state, transition

    # jax.debug.print('collecting traj', ordered=True)

    # runner_state, traj_batch = jax.lax.scan(
    #     _env_step, runner_state, None, config['NUM_STEPS']
    # )
    if config['NUM_RL_AGENTS'] > 0:
        runner_state.network_batteries.eval()
    if not config['USE_REC_RULE_BASED_POLICY']:
        runner_state.network_rec.eval()

    runner_state, traj_batch = nnx.scan(_env_step,
                                        in_axes=nnx.Carry,
                                        out_axes=(nnx.Carry, 0),
                                        length=config['NUM_STEPS'])(runner_state)

    _, transition = _env_step(runner_state)

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

    rewards_batteries = traj_batch.rewards_batteries[..., :config['NUM_RL_AGENTS']]

    # print(traj_batch.rewards_batteries.shape)
    # print(rewards_batteries.shape)

    reward_rec = traj_batch.reward_rec

    assert rewards_batteries.shape[1] == config['NUM_ENVS']
    assert rewards_batteries.shape[2] == config['NUM_RL_AGENTS']

    # if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS']:
    #     rewards_batteries = (rewards_batteries - rewards_batteries.mean(axis=(0, 1), keepdims=True)) / (rewards_batteries.std(axis=(0, 1), keepdims=True) + 1e-8)
    #     reward_rec = (reward_rec - reward_rec.mean()) / (reward_rec.std() + 1e-8)

    if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_BATTERIES']:
        rewards_batteries = (rewards_batteries - rewards_batteries.mean(axis=(0, 1), keepdims=True)) / (rewards_batteries.std(axis=(0, 1), keepdims=True) + 1e-8)
    if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_REC']:
        reward_rec = (reward_rec - reward_rec.mean()) / (reward_rec.std() + 1e-8)

    if config['NUM_RL_AGENTS'] > 0:
        _, advantages_batteries = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val_batteries), last_val_batteries),
            (traj_batch.done_batteries[..., :config['NUM_RL_AGENTS']], traj_batch.values_batteries[..., :config['NUM_RL_AGENTS']], rewards_batteries),
            reverse=True,
            unroll=32,
        )
        targets_batteries = advantages_batteries + traj_batch.values_batteries

        # if 'NORMALIZE_ADVANTAGES_BATTERIES':
        #     advantages_batteries = (advantages_batteries - advantages_batteries.mean(axis=(0, 1), keepdims=True)) / (advantages_batteries.std(axis=(0, 1), keepdims=True) + 1e-8)
        # if 'NORMALIZE_TARGETS_BATTERIES':
        #     targets_batteries = (targets_batteries - targets_batteries.mean(axis=(0, 1), keepdims=True)) / (targets_batteries.std(axis=(0, 1), keepdims=True) + 1e-8)

    else:
        advantages_batteries = 0.
        targets_batteries = 0.

    if not config['USE_REC_RULE_BASED_POLICY']:
        _, advantages_rec = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val_rec), last_val_rec),
            (traj_batch.done_rec, traj_batch.value_rec, reward_rec),
            reverse=True,
            unroll=32,
        )
        targets_rec = advantages_rec + traj_batch.value_rec

        # if 'NORMALIZE_ADVANTAGES_REC':
        #     advantages_rec = (advantages_rec - advantages_rec.mean()) / (
        #                 advantages_rec.std() + 1e-8)
        # if 'NORMALIZE_TARGETS_REC':
        #     targets_rec = (targets_rec - targets_rec.mean()) / (
        #                 targets_rec.std() + 1e-8)

    else:
        advantages_rec = 0.
        targets_rec = 0.

    return ((advantages_batteries, advantages_rec),
            (targets_batteries, targets_rec))

    # return ((advantages_batteries, advantages_rec),
    #         (advantages_batteries + traj_batch.values_batteries, advantages_rec + traj_batch.value_rec))

class UpdateState(NamedTuple):
    network: Union[StackedActorCritic, StackedRecurrentActorCritic, RECActorCritic, RECRecurrentActorCritic]
    optimizer: Union[StackedOptimizer, nnx.Optimizer]
    traj_batch: dict
    advantages: jnp.array
    targets: jnp.array
    rng: jax.random.PRNGKey


def update_batteries_network(runner_state: RunnerState, traj_batch, advantages, targets, config):
    def _update_epoch(update_state: UpdateState):
        def _update_minbatch(net_and_optim, traj_batch, advantages, targets):
            network_batteries, optimizer_batteries = net_and_optim

            # print(jax.tree.map(lambda x: x.shape, batch_info))

            def _loss_fn_batteries(network, traj_batch, gae, targets):
                traj_batch_data = (traj_batch.obs_batteries,
                                   traj_batch.actions_batteries,
                                   traj_batch.values_batteries,
                                   traj_batch.log_prob_batteries)
                traj_batch_obs, traj_batch_actions, traj_batch_values, traj_batch_log_probs = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_batch_data)
                traj_batch_obs, traj_batch_actions = jax.tree.map(lambda x: x[:config['NUM_RL_AGENTS']], (traj_batch_obs, traj_batch_actions))

                gae, targets = jax.tree.map(lambda x: x.swapaxes(0, 1), (gae, targets))

                # RERUN NETWORK
                pi, value = network(traj_batch_obs, obs_for_cri=traj_batch.obs_batteries)
                log_prob = pi.log_prob(traj_batch_actions)
                # print(log_prob.shape)

                if config['NORMALIZE_TARGETS_BATTERIES']:
                    targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch_values + (
                        value - traj_batch_values
                ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                # jax.debug.print('value losses {x}', x=value_losses.shape)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=1)

                assert value_loss.shape == (config['NUM_RL_AGENTS'],)

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch_log_probs)

                # jax.debug.print('ratio mean {x}, max {y}, min {z}, std {w}', x=ratio.mean(), y=ratio.max(), z=ratio.min(), w=ratio.std(), ordered=True)

                if config['NORMALIZE_ADVANTAGES_BATTERIES']:
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
                        - config['ENT_COEF_BATTERIES'] * entropy
                )

                assert total_loss.ndim == 1
                # jax.debug.print('bat_loss', ordered=True)

                total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
                return total_loss, (value_loss, loss_actor, entropy)

            def _loss_fn_batteries_recurrent(network, traj_batch:Transition, gae, targets):

                obs_battery_first = jax.tree.map(lambda x : x.swapaxes(0, 1)[:config['NUM_RL_AGENTS']], traj_batch.obs_batteries)
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
                                                       unroll=16)((network,) + jax.tree.map(lambda x: x[0], (traj_batch.lstm_states_prev_batteries.act_state, traj_batch.lstm_states_prev_batteries.cri_state)),
                                                                  data_for_network_act, data_for_network_cri, traj_batch.done_prev_batteries[:, :config['NUM_RL_AGENTS']])

                pi = network.apply_act_mlp(data_for_network_act, act_outputs)
                values = network.apply_cri_mlp(data_for_network_cri, cri_outputs)
                log_prob = pi.log_prob(traj_batch.actions_batteries[:, :config['NUM_RL_AGENTS']].swapaxes(0, 1))

                values_time_first = values.swapaxes(0, 1)
                log_prob_time_first = log_prob.swapaxes(0, 1)

                if config['NORMALIZE_TARGETS_BATTERIES']:
                    targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.values_batteries + (
                        values_time_first - traj_batch.values_batteries
                ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                value_losses = jnp.square(values_time_first - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                # jax.debug.print('value losses {x}', x=value_losses.shape)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=0)

                assert value_loss.shape == (config['NUM_RL_AGENTS'],)

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob_time_first - traj_batch.log_prob_batteries)

                if config['NORMALIZE_ADVANTAGES_BATTERIES']:
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

            total_loss_batteries, grads_batteries = grad_fn_batteries(
                network_batteries,
                traj_batch,
                advantages,
                targets
            )
            # jax.debug.print('gggg {x}', x=grads_batteries.log_std)
            # print('tot loss batteries')
            # print(jax.tree.map(lambda x: x.shape, total_loss_batteries))
            # jax.debug.print('bat {x}', x=total_loss_batteries[0])
            #
            # jax.debug.print('{x}', x=optax.global_norm(grads_batteries), ordered=True)
            # jax.debug.print('{x}', x=total_loss_batteries[1], ordered=True)
            # jax.debug.print('{x}', x=network_batteries.log_std.value, ordered=True)

            optimizer_batteries.update(grads_batteries)

            total_loss = total_loss_batteries

            return (network_batteries, optimizer_batteries), total_loss

        # train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(update_state.rng)
        batch_size = config['MINIBATCH_SIZE_BATTERIES'] * config['NUM_MINIBATCHES_BATTERIES']
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
                lambda x: x.reshape((x.shape[0],) + (-1, config['MINIBATCH_SIZE_BATTERIES']) + x.shape[2:]), batch
            )
            sequences = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )
            permutation = jax.random.permutation(_rng, config['NUM_MINIBATCHES_BATTERIES'])
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
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [config['NUM_MINIBATCHES_BATTERIES'], -1] + list(x.shape[1:])
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

    runner_state.network_batteries.train()

    update_state = UpdateState(network=runner_state.network_batteries, optimizer=runner_state.optimizer_batteries,
                               traj_batch=traj_batch, advantages=advantages, targets=targets, rng=runner_state.rng)

    # update_state = (runner_state.train_state, traj_batch, advantages, targets, runner_state.rng)

    scanned_update_epoch = nnx.scan(_update_epoch,
                                    in_axes=nnx.Carry,
                                    out_axes=(nnx.Carry, 0),
                                    length=config['UPDATE_EPOCHS'])

    update_state, loss_info = scanned_update_epoch(update_state)

    # update_state, loss_info = jax.lax.scan(
    #     _update_epoch, update_state, None, config['UPDATE_EPOCHS']
    # )

    # jax.debug.print('bat loss {x}', x=loss_info[0])

    runner_state.network_batteries.eval()

    runner_state = runner_state._replace(rng=update_state.rng)

    return runner_state, loss_info


def update_rec_network(runner_state, traj_batch, advantages, targets, config):
    def _update_epoch(update_state: UpdateState):
        def _update_minbatch(net_and_optim, traj_batch, advantages, targets):
            network_rec, optimizer_rec = net_and_optim

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

                if config['NORMALIZE_TARGETS_REC']:
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

                if config['NORMALIZE_ADVANTAGES_REC']:
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
                        - config['ENT_COEF_REC'] * entropy
                )
                # jax.debug.print('rec_loss', ordered=True)
                return total_loss, (value_loss, loss_actor, entropy)

            def _loss_fn_rec_recurrent(network, traj_batch:Transition, gae, targets):
                # RERUN NETWORK

                data_for_network_act, data_for_network_cri = network.prepare_data(traj_batch.obs_rec)

                def forward_pass_lstm(carry, data_act, data_cri, beginning):
                    network, act_state, cri_state = carry

                    init_states = network.get_initial_lstm_state()

                    act_state, cri_state = jax.lax.cond(beginning, lambda: init_states, lambda: (act_state, cri_state))
                    act_state, act_output = network.apply_lstm_act(data_act, act_state)
                    cri_state, cri_output = network.apply_lstm_cri(data_cri, cri_state)
                    return (network, act_state, cri_state), act_output, cri_output

                _, act_outputs, cri_outputs = nnx.scan(forward_pass_lstm,
                                                       in_axes=(nnx.Carry, 0, 0, 0),
                                                       out_axes=(nnx.Carry, 0, 0),
                                                       unroll=16)((network,) + jax.tree.map(lambda x: x[0], (traj_batch.lstm_states_prev_rec.act_state, traj_batch.lstm_states_prev_rec.cri_state)),
                                                                  data_for_network_act, data_for_network_cri, traj_batch.done_prev_rec)

                pi = network.apply_act_mlp(data_for_network_act, act_outputs)
                values, _ = network.apply_cri_mlp(data_for_network_cri, cri_outputs)

                log_prob = pi.log_prob(traj_batch.actions_rec + 1e-8)

                if config['NORMALIZE_TARGETS_REC']:
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

                if config['NORMALIZE_ADVANTAGES_REC']:
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                loss_actor1 = ratio * gae
                loss_actor2 = jnp.clip(ratio, 1.0 - config['CLIP_EPS'], 1.0 + config['CLIP_EPS']) * gae
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = loss_actor + config['VF_COEF'] * value_loss - config['ENT_COEF_REC'] * entropy

                # jax.debug.print('{t}', t=total_loss, ordered=True)

                return total_loss, (value_loss, loss_actor, entropy)

            # traj_data_batteries_for_loss = jax.tree.map(lambda x: x.swapaxes(0, 1), traj_data_batteries_for_loss)

            # print('traj_data_batteries_for_loss')
            # print(jax.tree.map(lambda x: x.shape, traj_data_batteries_for_loss))
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

            optimizer_rec.update(grads_rec)

            return (network_rec, optimizer_rec), total_loss_rec

        # train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(update_state.rng)
        batch_size = config['MINIBATCH_SIZE_REC'] * config['NUM_MINIBATCHES_REC']
        assert (
                batch_size == config['NUM_STEPS'] * config['NUM_ENVS']
        ), 'batch size must be equal to number of steps * number of envs'

        batch = (traj_batch, advantages, targets)

        # print(jax.tree.map(lambda x: x.shape, traj_batch))

        # print(jax.tree.map(lambda x: x.shape, batch))

        # jax.debug.print('bef {z}', z=jax.tree.map(lambda l: l.shape, traj_batch), ordered=True)

        if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
            batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1), batch
            )
            batch = jax.tree.map(
                lambda x: x.reshape((x.shape[0],) + (-1, config['MINIBATCH_SIZE_REC']) + x.shape[2:]), batch
            )
            sequences = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )
            permutation = jax.random.permutation(_rng, config['NUM_MINIBATCHES_REC'])
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
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [config['NUM_MINIBATCHES_REC'], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
        # print(jax.tree.map(lambda x: x.shape, minibatches))
        # jax.debug.print('aft3 {z}', z=jax.tree.map(lambda l: l.shape, minibatches[0]), ordered=True)

        scanned_update_minibatch = nnx.scan(_update_minbatch,
                                            in_axes=((nnx.Carry, 0, 0, 0)))

        _, total_loss = scanned_update_minibatch((update_state.network, update_state.optimizer), *minibatches)
        #
        # train_state, total_loss = jax.lax.scan(
        #     _update_minbatch, train_state, minibatches
        # )
        update_state = update_state._replace(rng=rng)
        return update_state, total_loss

    runner_state.network_rec.train()

    update_state = UpdateState(network=runner_state.network_rec, optimizer=runner_state.optimizer_rec,
                               traj_batch=traj_batch, advantages=advantages, targets=targets, rng=runner_state.rng)

    scanned_update_epoch = nnx.scan(_update_epoch,
                                    in_axes=nnx.Carry,
                                    out_axes=(nnx.Carry, 0),
                                    length=config['UPDATE_EPOCHS'])

    update_state, loss_info = scanned_update_epoch(update_state)

    # update_state, loss_info = jax.lax.scan(
    #     _update_epoch, update_state, None, config['UPDATE_EPOCHS']
    # )

    # jax.debug.print('rec loss {x}', x=loss_info[0])

    runner_state = runner_state._replace(rng=update_state.rng)

    runner_state.network_rec.eval()

    return runner_state, loss_info



# @partial(jax.jit, static_argnums=(0, 2, 3, 6))
def test_networks(env:RECEnv, train_state:TrainState, num_iter, config, rng, rec_rule_based_policy, curr_iter=0, print_data=False):

    networks_batteries, _, network_rec, _ = nnx.merge(train_state.graph_def, train_state.state)

    networks_batteries.eval()
    if not config['USE_REC_RULE_BASED_POLICY']:
        network_rec.eval()

    rng, _rng = jax.random.split(rng)

    obsv, env_state = env.reset(_rng, profile_index=0)

    if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic' and config['NUM_RL_AGENTS'] > 0:
        init_act_state_batteries, init_cri_state_batteries = networks_batteries.get_initial_lstm_state()
        act_state_batteries, cri_state_batteries = init_act_state_batteries, init_cri_state_batteries
    else:
        act_state_batteries, cri_state_batteries = None, None

    if not config['USE_REC_RULE_BASED_POLICY'] and config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
        init_act_state_rec, init_cri_state_rec = network_rec.get_initial_lstm_state()
        act_state_rec, cri_state_rec = init_act_state_rec, init_cri_state_rec
    else:
        act_state_rec, cri_state_rec = None, None

    # @scan_tqdm(num_iter, print_rate=num_iter // 100)
    def _env_step(runner_state, unused):
        obsv_batteries, env_state, act_state_batteries, cri_state_batteries, act_state_rec, cri_state_rec, rng, next_profile_index = runner_state

        # print('aaaaa', obsv_batteries[:config['NUM_RL_AGENTS']].shape)

        actions_batteries = []

        if config['NUM_RL_AGENTS'] > 0:

            obsv_batteries_rl = jax.tree.map(lambda x: x[:config['NUM_RL_AGENTS']], obsv_batteries)

            if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
                pi, value_batteries, act_state_batteries, cri_state_batteries = networks_batteries(obsv_batteries_rl, act_state_batteries, cri_state_batteries)
            else:
                pi, value_batteries = networks_batteries(obsv_batteries_rl, obs_for_cri=obsv_batteries)

            #deterministic action
            actions_batteries_rl = pi.mode()

            # print('act 1', actions_batteries_rl.shape)
            actions_batteries_rl = actions_batteries_rl.squeeze(axis=-1)
            actions_batteries.append(actions_batteries_rl)


        if config['NUM_BATTERY_FIRST_AGENTS'] > 0:
            idx_start_bf = config['NUM_RL_AGENTS']
            idx_end_bf = config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS']

            demand = obsv_batteries['demand'][idx_start_bf:idx_end_bf]
            generation = obsv_batteries['generation'][idx_start_bf:idx_end_bf]

            actions_batteries_battery_first = (generation - demand) / env_state.battery_states.electrical_state.v[idx_start_bf:idx_end_bf]

            actions_batteries.append(actions_batteries_battery_first)

        if config['NUM_ONLY_MARKET_AGENTS'] > 0:
            actions_batteries_only_market = jnp.zeros(
                (config['NUM_ONLY_MARKET_AGENTS'],))
            actions_batteries.append(actions_batteries_only_market)

        if config['NUM_RANDOM_AGENTS'] > 0:
            rng, _rng = jax.random.split(rng)

            actions_batteries_random = jax.random.uniform(_rng,
                                                          shape=(config['NUM_RANDOM_AGENTS'],),
                                                          minval=-1.,
                                                          maxval=1.)

            actions_batteries_random *= config['MAX_ACTION_RANDOM_AGENTS']

            actions_batteries.append(actions_batteries_random)

        actions_batteries = jnp.concat(actions_batteries, axis=0)


        actions_first = {env.battery_agents[i]: actions_batteries[i] for i in range(env.num_battery_agents)}
        actions_first[env.rec_agent] = jnp.zeros(env.num_battery_agents)

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward_first, done_first, info_first = env.step(
            _rng, env_state, actions_first
        )

        rec_obsv = obsv[env.rec_agent]

        if config.get('BATTERY_VALUES_IN_REC_OBS', False) or config.get('BATTERY_VALUES_IN_REC_OBS_CRI', False):
            if config['NUM_RL_AGENTS'] < config['NUM_BATTERY_AGENTS']:
                values_batteries_for_rec = jnp.concat((value_batteries, jnp.zeros(config['NUM_BATTERY_AGENTS'] - config['NUM_RL_AGENTS'])))
            else:
                values_batteries_for_rec = value_batteries

            rec_obsv['battery_values'] = values_batteries_for_rec

        if not config['USE_REC_RULE_BASED_POLICY']:
            if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
                pi, _, act_state_rec, cri_state_rec, separate_cri = network_rec(rec_obsv, act_state_rec, cri_state_rec, return_separate_cri=True)
            else:
                pi, _, separate_cri = network_rec(rec_obsv, return_separate_cri=True)
            actions_rec = pi.mean()
        else:
            actions_rec = rec_rule_based_policy(rec_obsv)

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

        obs_batteries = jax.tree.map(lambda *vals: jnp.stack(vals), *[obsv[a] for a in env.battery_agents])

        if config.get('REC_VALUE_IN_BATTERY_OBS', False) or config.get('REC_VALUE_IN_BATTERY_OBS_CRI', False):
            obs_batteries['rec_value'] = separate_cri

        runner_state = (obs_batteries, env_state, act_state_batteries, cri_state_batteries, act_state_rec, cri_state_rec, rng, next_profile_index)
        return runner_state, info

    obsv_batteries = jax.tree.map(lambda *vals: jnp.stack(vals), *[obsv[a] for a in env.battery_agents])

    if config.get('REC_VALUE_IN_BATTERY_OBS', False) or config.get('REC_VALUE_IN_BATTERY_OBS_CRI', False):
        obsv_batteries['rec_value'] = jnp.zeros(config['NUM_BATTERY_AGENTS'])

    runner_state = (obsv_batteries, env_state, act_state_batteries, cri_state_batteries, act_state_rec, cri_state_rec, rng, 1)

    runner_state, info = jax.lax.scan(_env_step, runner_state, jnp.arange(num_iter))

    reward_type = 'pure_reward'

    if print_data:
        # if config['NUM_RL_AGENTS'] > 0:
        #     jax.debug.print('curr_iter: {i}\n\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}\n\tr_glob: {r_glob}\n\tr_rec: {r_rec}\n'
        #                     '\tmean soc: {mean_soc}\n\tstd actions: {std_act}\n\tself consumption: {sc}\n',
        #                     i=curr_iter, r_tot=jnp.sum(info['r_tot'][:, :config['NUM_RL_AGENTS']]), r_trad=jnp.sum(info[reward_type]['r_trad'][:, :config['NUM_RL_AGENTS']]),
        #                     r_deg=jnp.sum(info[reward_type]['r_deg'][:, :config['NUM_RL_AGENTS']]), r_clip=jnp.sum(info[reward_type]['r_clipping'][:, :config['NUM_RL_AGENTS']]),
        #                     r_glob=jnp.sum(info[reward_type]['r_glob'][:, :config['NUM_RL_AGENTS']]), r_rec=jnp.sum(info['rec_reward']),
        #                     mean_soc=jnp.mean(info['soc'][:, :config['NUM_RL_AGENTS']]), std_act=jnp.std(info['actions_batteries'], axis=0),
        #                     sc=jnp.sum(info['self_consumption']))
        # jax.debug.print('rule based:')
        # jax.debug.print(
        #     'curr_iter: {i}\n\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}\n\tr_glob: {r_glob}\n\tr_rec: {r_rec}\n'
        #     '\tmean soc: {mean_soc}\n\tstd actions: {std_act}\n\tself consumption: {sc}\n',
        #     i=curr_iter, r_tot=jnp.sum(info['r_tot'][:, config['NUM_RL_AGENTS']:]),
        #     r_trad=jnp.sum(info[reward_type]['r_trad'][:, config['NUM_RL_AGENTS']:]),
        #     r_deg=jnp.sum(info[reward_type]['r_deg'][:, config['NUM_RL_AGENTS']:]),
        #     r_clip=jnp.sum(info[reward_type]['r_clipping'][:, config['NUM_RL_AGENTS']:]),
        #     r_glob=jnp.sum(info[reward_type]['r_glob'][:, config['NUM_RL_AGENTS']:]), r_rec=jnp.sum(info['rec_reward']),
        #     mean_soc=jnp.mean(info['soc'][:, config['NUM_RL_AGENTS']:]),
        #     std_act=jnp.std(info['actions_batteries'], axis=0),
        #     sc=jnp.sum(info['self_consumption']))

        jax.debug.print('curr_iter: {i}', i=curr_iter)
        for i in range(config['NUM_BATTERY_AGENTS']):
            jax.debug.print(
                '\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}\n\tr_glob: {r_glob}\n\tr_rec: {r_rec}\n'
                '\tmean soc: {mean_soc}\n\tstd actions: {std_act}\n\tself consumption: {sc}\n',
                r_tot=jnp.sum(info['r_tot'][:, i]),
                r_trad=jnp.sum(info[reward_type]['r_trad'][:, i]),
                r_deg=jnp.sum(info[reward_type]['r_deg'][:, i]),
                r_clip=jnp.sum(info[reward_type]['r_clipping'][:, i]),
                r_glob=jnp.sum(info[reward_type]['r_glob'][:, i]), r_rec=jnp.sum(info['rec_reward']),
                mean_soc=jnp.mean(info['soc'][:, i]),
                std_act=jnp.std(info['actions_batteries'], axis=0),
                sc=jnp.sum(info['self_consumption']))

        jax.debug.print('\n\tr_tot: {x}', x=jnp.sum(info['r_tot'][:, :config['NUM_RL_AGENTS']]))

    return info