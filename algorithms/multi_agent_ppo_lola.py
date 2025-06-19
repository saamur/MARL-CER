from time import time
from datetime import datetime

import jax
import jax.numpy as jnp
from jax.experimental import io_callback
import flax.nnx as nnx
import optax

from algorithms.wrappers import VecEnvJaxMARL

from algorithms.train_core import StackedOptimizer, ValidationLogger, TrainState, config_enhancer, networks_builder, schedule_builder, optimizer_builder, prepare_runner_state
from algorithms.train_core import update_rec_network_lola
from algorithms.train_core import test_networks

from algorithms.tqdm_custom import scan_tqdm as tqdm_custom
from ernestogym.envs.multi_agent.env import RECEnv
import algorithms.utils as utils


def make_train(config, env:RECEnv, network_batteries=None, network_rec=None, seed=123):

    print('PPO LOLA')

    config['NUM_MINIBATCHES_REC'] = 1
    config['NUM_EPOCHS_REC'] = 1

    config_enhancer(config, env, is_rec_ppo=True)

    del config['NUM_MINIBATCHES_REC']
    del config['NUM_EPOCHS_REC']

    env = VecEnvJaxMARL(env)

    network_batteries, network_rec = networks_builder(config, network_batteries, network_rec, seed)

    schedule_batteries = schedule_builder(config['LR_SCHEDULE_BATTERIES'],
                                          config['LR_BATTERIES'],
                                          config['NUM_ITERATIONS'] * config['NUM_EPOCHS_BATTERIES'] * config['NUM_MINIBATCHES_BATTERIES'],
                                          lr_end=config.get('LR_BATTERIES_MIN', 0.),
                                          frac_dynamic=config.get('FRACTION_DYNAMIC_LR_BATTERIES', 1.),
                                          frac_warmup=config.get('WARMUP_SCHEDULE_BATTERIES', 0.),
                                          )

    optimizer_batteries = optimizer_builder(config['OPTIMIZER_BATTERIES'], schedule_batteries,
                                            beta_adam=config.get('BETA_ADAM_BATTERIES', 0.9),
                                            momentum=config.get('MOMENTUM_BATTERIES', None))

    tx_bat = optax.chain(optax.clip_by_global_norm(config['MAX_GRAD_NORM']), optimizer_batteries)
    optimizer_batteries = StackedOptimizer(config['NUM_RL_AGENTS'], network_batteries, tx_bat)

    schedule_rec = schedule_builder(config['LR_SCHEDULE_REC'],
                                    config['LR_REC'],
                                    config['NUM_ITERATIONS'],
                                    lr_end=config.get('LR_REC_MIN', 0.),
                                    frac_dynamic=config.get('FRACTION_DYNAMIC_LR_REC', 1.),
                                    frac_warmup=config.get('WARMUP_SCHEDULE_REC', 0.),
                                    )
    optimizer_rec = optimizer_builder(config['OPTIMIZER_REC'], schedule_rec,
                                      beta_adam=config.get('BETA_ADAM_REC', 0.9),
                                      momentum=config.get('MOMENTUM_REC', None))

    tx_rec = optax.chain(optax.clip_by_global_norm(config['MAX_GRAD_NORM']), optimizer_rec)

    optimizer_rec = nnx.Optimizer(network_rec, tx_rec)

    return env, network_batteries, optimizer_batteries, network_rec, optimizer_rec

# @partial(nnx.jit, static_argnums=(0, 1, 7, 8, 9, 11))
def train(env: RECEnv, config, world_metadata, network_batteries, optimizer_batteries, network_rec, optimizer_rec, rng, validate=True, freq_val=None, val_env=None,
              val_rng=None, val_num_iters=None, path_saving=None):

    actual_num_iterations = int(config['NUM_ITERATIONS'] * config.get('TRUNCATE_FRACTION', 1))

    if validate:
        if freq_val is None or val_env is None or val_rng is None or val_num_iters is None:
            raise ValueError(
                "'freq_val', 'val_env', 'val_rng' and 'val_num_iters' must be defined when 'validate' is True")

    dir_name = (datetime.now().strftime('%Y%m%d_%H%M%S') + '/')
    directory = path_saving + dir_name
    logger = ValidationLogger(config, world_metadata, directory, actual_num_iterations, freq_val)

    def update_val_info(val_info, train_state):
        logger.log_val(val_info, train_state)

    @tqdm_custom(0, 0, 1, actual_num_iterations, print_rate=1)
    def _update_step(runner_state, curr_iter):

        runner_state = update_rec_network_lola(runner_state, env, config)

        if validate:
            train_state = TrainState(*nnx.split((runner_state.network_batteries, runner_state.network_rec)))

            jax.lax.cond(curr_iter % freq_val == 0,
                         lambda: io_callback(update_val_info,
                                             None,
                                             test_networks(val_env, train_state, val_num_iters, config, val_rng, curr_iter=curr_iter, print_data=True),
                                             train_state,
                                             ordered=True),
                         lambda: None)

        return runner_state

    if config['NUM_RL_AGENTS'] > 0:
        network_batteries.eval()
    network_rec.eval()

    runner_state = prepare_runner_state(env, config, network_batteries, optimizer_batteries, network_rec,
                                            optimizer_rec, rng)

    scanned_update_step = nnx.scan(_update_step,
                                   in_axes=(nnx.Carry, 0),
                                   out_axes=nnx.Carry)

    scanned_update_step = nnx.jit(scanned_update_step)

    runner_state = scanned_update_step(runner_state, jnp.arange(actual_num_iterations))

    if config['NUM_RL_AGENTS'] > 0:
        runner_state.network_batteries.eval()
    runner_state.network_rec.eval()

    print('Saving...')

    t0 = time()

    logger.save_final(runner_state.network_batteries, runner_state.network_rec)

    print(f'Saving time: {t0 - time():.2f} s')

    if validate:
        return  runner_state, logger.val_infos
    else:
        return runner_state
