import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

import jax
jax.config.update('jax_default_matmul_precision', 'float32')
import jax.numpy as jnp

from flax import nnx
from flax.core.frozen_dict import freeze, unfreeze

from ernestogym.envs.multi_agent.env import RECEnv
from ernestogym.envs.multi_agent.utils import get_world_metadata, get_world_metadata_from_template, get_world_data
from algorithms.multi_agent_ppo_lola_inspired import make_train, train

import time

print('a', os.getcwd())

os.chdir('../../..')
print('b', os.getcwd())


battery_type = 'degrading_dropflow'

def main():

    ##############################  1  ##############################

    world_metadata = get_world_metadata_from_template('3_agents_passive_plus_minus')
    train_params, test_params = get_world_data(world_metadata, get_train=True, get_test=True)

    num_envs = 4
    total_timesteps = 8760 * num_envs * 800

    config = {

        'TRUNCATE_FRACTION': 1.,
        'RESTORE_ENV_STATE_AFTER_REC_UPDATE': True,

        'NUM_CONSECUTIVE_ITERATIONS_BATTERIES': 1,
        'NUM_CONSECUTIVE_ITERATIONS_REC': 3,

        'NUM_RL_AGENTS': 3,
        'NUM_BATTERY_FIRST_AGENTS': 0,
        'NUM_ONLY_MARKET_AGENTS': 0,
        'NUM_RANDOM_AGENTS': 0,
        'MAX_ACTION_RANDOM_AGENTS': 2.,

        'LR_SCHEDULE_BATTERIES': 'cosine',
        'LR_BATTERIES': 5e-5,
        'LR_BATTERIES_MIN': 1e-7,
        'FRACTION_DYNAMIC_LR_BATTERIES': 1.,
        'FRACTION_WARMUP_SCHEDULE_BATTERIES': 0.,
        'OPTIMIZER_BATTERIES': 'adamw',
        'BETA_ADAM_BATTERIES': 0.9,

        'LR_SCHEDULE_REC': 'cosine',
        'LR_REC': 4e-4,
        'LR_REC_MIN': 1e-6,
        'FRACTION_DYNAMIC_LR_REC': 1.,
        'FRACTION_WARMUP_SCHEDULE_REC': 0.,
        'OPTIMIZER_REC': 'adamw',
        'BETA_ADAM_REC': 0.6,

        'NUM_ENVS': num_envs,
        'NUM_STEPS': 8192,
        'TOTAL_TIMESTEPS': total_timesteps,

        'NUM_EPOCHS_BATTERIES': 10,
        'NUM_MINIBATCHES_BATTERIES': 32,

        'NUM_STEPS_FOR_REC_UPDATE': 256,
        'NUM_MINIBATCHES_BATTERIES_FOR_REC_UPDATE': 2,
        'NUM_EPOCHS_BATTERIES_FOR_REC_UPDATE': 3,
        'UPDATE_TIMES_REC': 32,
        'LR_BATTERIES_FOR_REC_UPDATE': 1e-2,

        'GAMMA': 0.99,
        'GAE_LAMBDA': 0.98,
        'CLIP_EPS': 0.20,
        'VF_COEF': 0.5,
        'MAX_GRAD_NORM': 0.5,
        'ENT_COEF': 0.,

        'NETWORK_TYPE_BATTERIES': 'actor_critic',
        'NET_ARCH_BATTERIES': (64, 32),
        'NETWORK_TYPE_REC': 'mlp',
        'NON_SHARED_NET_ARCH_REC_AFTER': (64, 32),

        'ACTIVATION': 'tanh',

        'NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_BATTERIES': False,
        'NORMALIZE_TARGETS_BATTERIES': False,
        'NORMALIZE_ADVANTAGES_BATTERIES': True,

        'NORMALIZE_NN_INPUTS': True,

    }

    rng = jax.random.PRNGKey(42)
    val_rng = jax.random.PRNGKey(51)
    val_num_iters = 8760 * 5
    env_testing = RECEnv(test_params, battery_type)

    env = RECEnv(train_params, battery_type)
    env, networks_batteries, optimizer_batteries, network_rec, optimizer_rec = make_train(config, env)

    config = freeze(config)

    t0 = time.time()

    train(env, config, world_metadata, networks_batteries, optimizer_batteries, network_rec, optimizer_rec, rng,
          validate=True, freq_val=int(10 * (config['NUM_CONSECUTIVE_ITERATIONS_BATTERIES'] + config['NUM_CONSECUTIVE_ITERATIONS_REC'])/config['NUM_CONSECUTIVE_ITERATIONS_BATTERIES']),
          val_env=env_testing,
          val_rng=val_rng,
          val_num_iters=val_num_iters,
          path_saving='trained_agents/')

    print(f'time: {time.time() - t0:.2f} s')

if __name__ == '__main__':
    main()
