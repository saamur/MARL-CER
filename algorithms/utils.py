import os
from copy import deepcopy
from datetime import datetime
import orbax.checkpoint as ocp
import flax.nnx as nnx
import pickle

import algorithms.ppo as ppo
import algorithms.recurrent_ppo as recurrent_ppo

from jax_tqdm import scan_tqdm, loop_tqdm
from ernestogym.envs_jax.single_agent.env import MicroGridEnv
from functools import partial
import jax
import jax.numpy as jnp


path_base = '/media/samuele/Disco/PycharmProjectsUbuntu/MARL-CER/trained_agents/'

def construct_net_from_config(config, rng):
    if config['NETWORK'] == 'actor_critic':
        return ppo.ActorCritic(
            config["OBSERVATION_SPACE_SIZE"],
            config["ACTION_SPACE_SIZE"],
            activation=config["ACTIVATION"],
            net_arch=config.get("NET_ARCH"),
            act_net_arch=config.get("ACT_NET_ARCH"),
            cri_net_arch=config.get("CRI_NET_ARCH"),
            add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
            rngs=rng
        )
    elif config['NETWORK'] == 'recurrent_actor_critic':
        return recurrent_ppo.RecurrentActorCritic(
            config["OBSERVATION_SPACE_SIZE"],
            config["ACTION_SPACE_SIZE"],
            num_sequences=config["NUM_SEQUENCES"],
            activation=config["ACTIVATION"],
            lstm_activation=config["LSTM_ACTIVATION"],
            net_arch=config.get("NET_ARCH"),
            act_net_arch=config.get("ACT_NET_ARCH"),
            cri_net_arch=config.get("CRI_NET_ARCH"),
            lstm_net_arch=config.get("LSTM_NET_ARCH"),
            lstm_act_net_arch=config.get("LSTM_ACT_NET_ARCH"),
            lstm_cri_net_arch=config.get("LSTM_CRI_NET_ARCH"),
            add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
            rngs=rng
        )
    else:
        raise ValueError('Invalid network name')

def construct_net_from_config_multi_agent(config, rng):
    config = deepcopy(config)
    config['OBSERVATION_SPACE_SIZE'] = config['BATTERY_OBSERVATION_SPACE_SIZE']
    config['ACTION_SPACE_SIZE'] = config['BATTERY_ACTION_SPACE_SIZE']

    @nnx.split_rngs(splits=config['NUM_BATTERY_AGENTS'])
    @nnx.vmap(in_axes=(None, 0))
    def get_network(config, rng):
        return construct_net_from_config(config, rng)

    networks = get_network(config, rng)

    return networks


def save_state(network, config, params: dict, val_info:dict=None, env_type='normal', additional_info=''):
    dir_name = (datetime.now().strftime("%Y%m%d_%H%M%S") +
                '_lr_' + str(config['LR']) +
                '_tot_timesteps_' + str(config['TOTAL_TIMESTEPS']) +
                '_anneal_rl_' + str(config['ANNEAL_LR']) +
                '_' + env_type +
                '_' + config['NETWORK'])

    if additional_info != '':
        dir_name += '_' + additional_info

    os.makedirs(path_base + dir_name)

    _, state = nnx.split(network)

    with open(path_base + dir_name + '/state.pkl', 'wb') as file:
        pickle.dump(state, file)

    with open(path_base + dir_name + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)

    params = deepcopy(params)
    del params['demand']['data']
    if 'generation' in params.keys():
        del params['generation']['data']
    if 'market' in params.keys():
        del params['market']['data']
    if 'temp_ambient' in params.keys():
        del params['temp_ambient']['data']

    with open(path_base + dir_name + '/params.pkl', 'wb') as file:
        pickle.dump(params, file)

    with open(path_base + dir_name + '/val_info.pkl', 'wb') as file:
        pickle.dump(val_info, file)


def restore_state(path):

    with open(path + '/config.pkl', 'rb') as file:
        config = pickle.load(file)

    with open(path + '/params.pkl', 'rb') as file:
        params = pickle.load(file)

    network_shape = construct_net_from_config(config, nnx.Rngs(0))
    graphdef, abstract_state = nnx.split(network_shape)

    with open(path + '/state.pkl', 'rb') as file:
        state_restored = pickle.load(file)

    network = nnx.merge(graphdef, state_restored)

    with open(path + '/val_info.pkl', 'rb') as file:
        val_info = pickle.load(file)

    return network, config, params, val_info




# def save_state(network, config, params, env_type='normal', additional_info=''):
#     dir_name = (datetime.now().strftime("%Y%m%d_%H%M%S") +
#                 '_lr_' + str(config['LR']) +
#                 '_tot_timesteps_' + str(config['TOTAL_TIMESTEPS']) +
#                 '_anneal_rl_' + str(config['ANNEAL_LR']) +
#                 '_' + env_type)
#
#     if additional_info != '':
#         dir_name += '_' + additional_info
#
#     checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
#     _, state = nnx.split(network)
#     checkpointer.save(path_base + dir_name + '/state', state)
#
#     with open(path_base + dir_name + '/config.pkl', 'wb') as file:
#         pickle.dump(config, file)
#
#     with open(path_base + dir_name + '/params.pkl', 'wb') as file:
#         pickle.dump(params, file)
#
# def restore_state(path):
#
#     with open(path + '/config.pkl', 'rb') as file:
#         config = pickle.load(file)
#
#     with open(path + '/params.pkl', 'rb') as file:
#         params = pickle.load(file)
#
#     checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
#
#     network_shape = construct_net_from_config(config, nnx.Rngs(0))
#     graphdef, abstract_state = nnx.split(network_shape)
#     state_restored = checkpointer.restore(path + '/state', abstract_state)
#
#     network = nnx.merge(graphdef, state_restored)
#
#     return network, config, params