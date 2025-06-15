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
from typing import Sequence, NamedTuple, Any, Union, Dict
import distrax

from algorithms.wrappers import VecEnvJaxMARL

import algorithms.utils as utils
from ernestogym.envs.multi_agent.env import RECEnv, EnvState
from algorithms.networks import StackedActorCritic, StackedRecurrentActorCritic, RECActorCritic, RECRecurrentActorCritic, RECMLP
from algorithms.normalization_custom import RunningNormScalar

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

class ValidationLogger:

    def __init__(self, config, world_metadata, directory, num_iterations, freq_val):
        self.directory = directory
        self.config = config
        self.world_metadata = world_metadata
        self.num_iterations = num_iterations
        self.freq_val = freq_val
        self.val_infos = None

        self.i = 0

    def initialize(self, val_info, freq_val):
        self.val_infos = dict(
            jax.tree.map(lambda x: np.empty_like(x, shape=((self.num_iterations - 1) // freq_val + 1,) + x.shape),
                         val_info))
        self.i = 0


    def log_val(self, val_info, train_state):

        if self.val_infos is None:
            self.initialize(val_info, self.freq_val)

        def update(logs, new):
            logs[self.i] = new

        network_batteries, network_rec = nnx.merge(train_state.graph_def, train_state.state)

        val_info = jax.device_put(val_info, device=jax.devices('cpu')[0])
        jax.tree.map(update, self.val_infos, val_info)

        utils.save_state_multiagent(self.directory, network_batteries, network_rec, self.config, self.world_metadata,
                                    is_checkpoint=True, num_steps=self.i)

        self.i += 1

    def save_final(self, network_batteries, network_rec):
        utils.save_state_multiagent(self.directory, network_batteries, network_rec, self.config, self.world_metadata, None, self.val_infos, is_checkpoint=False)


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
    network_rec: Union[RECActorCritic, RECRecurrentActorCritic, RECMLP]
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

class UpdateState(NamedTuple):
    network: Union[StackedActorCritic, StackedRecurrentActorCritic, RECActorCritic, RECRecurrentActorCritic]
    optimizer: Union[StackedOptimizer, nnx.Optimizer]
    traj_batch: Transition
    advantages: jnp.array
    targets: jnp.array
    rng: jax.random.PRNGKey



def config_enhancer(config: Dict, env:RECEnv, is_rec_ppo):
    config['NUM_ITERATIONS'] = config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']


    if 'NUM_MINIBATCHES_BATTERIES' not in config.keys():
        if 'NUM_MINIBATCHES' not in config.keys():
            raise ValueError("At least one of config['NUM_MINIBATCHES_BATTERIES'] and config['NUM_MINIBATCHES'] must be provided")
        config['NUM_MINIBATCHES_BATTERIES'] = config['NUM_MINIBATCHES']

    if 'NUM_EPOCHS_BATTERIES' not in config.keys():
        if 'NUM_EPOCHS' not in config.keys():
            raise ValueError("At least one of config['NUM_EPOCHS_BATTERIES'] and config['NUM_EPOCHS'] must be provided")
        config['NUM_EPOCHS_BATTERIES'] = config['NUM_EPOCHS']

    config['MINIBATCH_SIZE_BATTERIES'] = config['NUM_ENVS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES_BATTERIES']

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

    if 'GAMMA_BATTERIES' not in config.keys():
        if 'GAMMA' not in config.keys():
            raise ValueError("At least one of config['GAMMA_BATTERIES'] and config['GAMMA'] must be provided")
        config['GAMMA_BATTERIES'] = config['GAMMA']

    config['BATTERY_ACTION_SPACE_SIZE'] = env.action_space(env.battery_agents[0]).shape[0]
    config['BATTERY_OBS_KEYS'] = tuple(env.obs_battery_agents_keys)
    config['BATTERY_OBS_IS_SEQUENCE'] = env.obs_is_sequence_battery
    config['BATTERY_OBS_IS_NORMALIZABLE'] = env.obs_is_normalizable_battery


    if is_rec_ppo:
        if 'NUM_MINIBATCHES_REC' not in config.keys():
            if 'NUM_MINIBATCHES' not in config.keys():
                raise ValueError("At least one of config['NUM_MINIBATCHES_REC'] and config['NUM_MINIBATCHES'] must be provided")
            config['NUM_MINIBATCHES_REC'] = config['NUM_MINIBATCHES']

        if 'NUM_EPOCHS_REC' not in config.keys():
            if 'NUM_EPOCHS' not in config.keys():
                raise ValueError(
                    "At least one of config['NUM_EPOCHS_REC'] and config['NUM_EPOCHS'] must be provided")
            config['NUM_EPOCHS_REC'] = config['NUM_EPOCHS']

        config['MINIBATCH_SIZE_REC'] = config['NUM_ENVS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES_REC']

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

        if 'GAMMA_REC' not in config.keys():
            if 'GAMMA' not in config.keys():
                raise ValueError("At least one of config['GAMMA_REC'] and config['GAMMA'] must be provided")
            config['GAMMA_REC'] = config['GAMMA']


    config['REC_ACTION_SPACE_SIZE'] = env.action_space(env.rec_agent).shape[0]
    config['REC_OBS_KEYS'] = tuple(env.obs_rec_keys)
    config['NUM_BATTERY_AGENTS'] = env.num_battery_agents
    config['PASSIVE_HOUSES'] = (env.num_passive_houses > 0)
    config['REC_OBS_IS_SEQUENCE'] = env.obs_is_sequence_rec
    config['REC_OBS_IS_LOCAL'] = env.obs_is_local_rec
    config['REC_OBS_IS_NORMALIZABLE'] = env.obs_is_normalizable_rec

    assert (len(env.battery_agents) ==
            config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS'] +
            config['NUM_ONLY_MARKET_AGENTS'] + config['NUM_RANDOM_AGENTS'])

def schedule_builder(name, lr_init, tot_updates, lr_end=0., frac_dynamic=1., frac_warmup=0.) -> optax.Schedule:
    tot_updates = int(tot_updates)
    warmup_steps = int(tot_updates * frac_warmup)
    dynamic_steps = int(tot_updates * frac_dynamic)

    if name == 'linear':
        return optax.schedules.linear_schedule(lr_init, lr_end, dynamic_steps-warmup_steps, warmup_steps)
    elif name == 'cosine':
        return optax.schedules.warmup_cosine_decay_schedule(0., lr_init, warmup_steps, dynamic_steps, lr_end)
    elif name == 'constant' or name == 'const':
        return optax.schedules.constant_schedule(lr_init)
    else:
        raise ValueError(f'Unknown schedule: {name}')

def optimizer_builder(name, scheduler, beta_adam=0.9, momentum=None):
    if name == 'adam':
        return optax.adam(learning_rate=scheduler, b1=beta_adam, eps=0., eps_root=1e-10)
    elif name == 'adamw':
        return optax.adamw(learning_rate=scheduler, b1=beta_adam, eps=0., eps_root=1e-10)
    elif name == 'sgd':
        return optax.sgd(learning_rate=scheduler, momentum=momentum)
    elif name == 'rmsprop':
        return optax.rmsprop(learning_rate=scheduler, momentum=momentum)
    else:
        raise ValueError("Optimizer '{}' not recognized".format(name))

def networks_builder(config, network_batteries=None, network_rec=None, seed=123):
    key = jax.random.PRNGKey(seed)
    key, _key = jax.random.split(key)
    rng = nnx.Rngs(_key)

    if network_batteries is None:
        network_batteries = utils.construct_battery_net_from_config_multi_agent(config, rng,
                                                                                num_nets=config['NUM_RL_AGENTS'])

    key, _key = jax.random.split(key)
    rng = nnx.Rngs(_key)

    if network_rec is None and not config.get('USE_REC_RULE_BASED_POLICY', False):
        network_rec = utils.construct_rec_net_from_config_multi_agent(config, rng)

    return network_batteries, network_rec

def trainnnn(update_step, runner_state, env:RECEnv, config, network_batteries, optimizer_batteries, network_rec, optimizer_rec,
                  rng, world_metadata, validate=True, freq_val=None, val_env=None, val_rng=None,
                  val_num_iters=None, path_saving=None):

    infos = {}
    val_infos = {}

    dir_name = (datetime.now().strftime('%Y%m%d_%H%M%S') + '/')

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

        network_batteries, _, network_rec, _ = nnx.merge(train_state.graph_def, train_state.state)

        val_info = jax.device_put(val_info, device=jax.devices('cpu')[0])
        jax.tree.map(update, val_infos, val_info)

        utils.save_state_multiagent(directory, network_batteries, network_rec, config, world_metadata, is_checkpoint=True, num_steps=i)

    scanned_update_step = nnx.scan(nnx.jit(update_step),
                                   in_axes=(nnx.Carry, 0),
                                   out_axes=nnx.Carry)

    runner_state = scanned_update_step(runner_state, jnp.arange(config['NUM_UPDATES']))

def prepare_runner_state(env: RECEnv, config, network_batteries, optimizer_batteries, network_rec, optimizer_rec, rng):

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

    if not config.get('USE_REC_RULE_BASED_POLICY', False) and config.get('NETWORK_TYPE_REC', 'na') == 'recurrent_actor_critic':
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

    obsv_batteries = jax.tree.map(lambda *vals: jnp.stack(vals, axis=1), *[obsv[a] for a in env.battery_agents])

    runner_state = RunnerState(network_batteries=network_batteries,
                               network_rec=network_rec,
                               optimizer_batteries=optimizer_batteries,
                               optimizer_rec=optimizer_rec,
                               env_state=env_state,
                               last_obs_batteries=obsv_batteries,
                               rng=rng,
                               done_prev_batteries=episode_starts_batteries,
                               done_prev_rec=episode_starts_rec,
                               last_lstm_state_batteries=lstm_state_batteries,
                               last_lstm_state_rec=lstm_state_rec)

    return runner_state
