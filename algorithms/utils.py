import os
from copy import deepcopy
from datetime import datetime
import orbax.checkpoint as ocp
import flax.nnx as nnx
import pickle

import algorithms.ppo as ppo
import algorithms.recurrent_ppo as recurrent_ppo
import algorithms.multi_agent_ppo as multi_agent_ppo

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

def construct_battery_net_from_config_multi_agent(config, rng):
    config = deepcopy(config)
    config['OBSERVATION_SPACE_SIZE'] = config['BATTERY_OBSERVATION_SPACE_SIZE']
    config['ACTION_SPACE_SIZE'] = config['BATTERY_ACTION_SPACE_SIZE']

    @nnx.split_rngs(splits=config['NUM_BATTERY_AGENTS'])
    def thing(rng):
        return multi_agent_ppo.WrappedStackedActorCritic(
            config["OBSERVATION_SPACE_SIZE"],
            config["ACTION_SPACE_SIZE"],
            activation=config["ACTIVATION"],
            net_arch=config.get("NET_ARCH"),
            act_net_arch=config.get("ACT_NET_ARCH"),
            cri_net_arch=config.get("CRI_NET_ARCH"),
            add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
            rngs=rng
        )
    return thing(rng)

def construct_rec_net_from_config_multi_agent(config, rng):
    return multi_agent_ppo.RECActorCritic(config["REC_INPUT_NETWORK_SIZE"],
                                          config['NUM_BATTERY_AGENTS'],
                                          config['ACTIVATION'],
                                          rngs=rng,
                                          net_arch=config.get("NET_ARCH"),
                                          act_net_arch=config.get("ACT_NET_ARCH"),
                                          cri_net_arch=config.get("CRI_NET_ARCH"),
                                          passive_houses=config['PASSIVE_HOUSES'])

# def construct_net(network_type:str, input_size:int, output_size:int, activation:str, rng,
#                   net_arch:list=None, act_net_arch:list=None, cri_net_arch:list=None,
#                   add_logistic_to_actor:bool=False, num_sequences:int=None,
#                   lstm) -> nnx.Module:
#     if network_type == 'actor_critic':
#         return ppo.ActorCritic(
#             input_size,
#             output_size,
#             activation=activation,
#             net_arch=net_arch,
#             act_net_arch=act_net_arch,
#             cri_net_arch=cri_net_arch,
#             add_logistic_to_actor=add_logistic_to_actor,
#             rngs=rng
#         )
#     elif network_type == 'recurrent_actor_critic':
#         assert num_sequences is not None
#         return recurrent_ppo.RecurrentActorCritic(
#             input_size,
#             output_size,
#             num_sequences=num_sequences,
#             activation=activation,
#             lstm_activation=config["LSTM_ACTIVATION"],
#             net_arch=net_arch,
#             act_net_arch=act_net_arch,
#             cri_net_arch=cri_net_arch,
#             lstm_net_arch=config.get("LSTM_NET_ARCH"),
#             lstm_act_net_arch=config.get("LSTM_ACT_NET_ARCH"),
#             lstm_cri_net_arch=config.get("LSTM_CRI_NET_ARCH"),
#             add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
#             rngs=rng
#         )
#     else:
#         raise ValueError('Invalid network name')


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

def save_state_multiagent(networks_batteries, network_rec, config, params: dict, val_info:dict=None, env_type='normal', additional_info=''):
    dir_name = (datetime.now().strftime("%Y%m%d_%H%M%S") +
                '_lr_' + str(config['LR']) +
                '_tot_timesteps_' + str(config['TOTAL_TIMESTEPS']) +
                '_anneal_rl_' + str(config['ANNEAL_LR']) +
                '_' + env_type +
                '_multiagent')

    if additional_info != '':
        dir_name += '_' + additional_info

    os.makedirs(path_base + dir_name)

    _, state_batteries = nnx.split(networks_batteries)
    _, state_rec = nnx.split(network_rec)

    with open(path_base + dir_name + '/state_batteries.pkl', 'wb') as file:
        pickle.dump(state_batteries, file)
    with open(path_base + dir_name + '/state_rec.pkl', 'wb') as file:
        pickle.dump(state_rec, file)


    with open(path_base + dir_name + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)

    params = deepcopy(params)

    for record in (params['demands_battery_houses'] + params['demands_passive_houses'] +
                   params['generations_battery_houses'] + params['generations_passive_houses'] +
                   params['selling_prices_battery_houses'] + params['selling_prices_passive_houses'] +
                   params['buying_prices_battery_houses'] + params['buying_prices_passive_houses'] +
                   params['temp_amb_battery_houses']):
        if 'data' in record.keys():
            del record['data']

    with open(path_base + dir_name + '/params.pkl', 'wb') as file:
        pickle.dump(params, file)

    with open(path_base + dir_name + '/val_info.pkl', 'wb') as file:
        pickle.dump(val_info, file)


def restore_state_multi_agent(path):

    with open(path + '/config.pkl', 'rb') as file:
        config = pickle.load(file)

    with open(path + '/params.pkl', 'rb') as file:
        params = pickle.load(file)

    network_batteries_shape = construct_battery_net_from_config_multi_agent(config, nnx.Rngs(0))
    graphdef_batteries, abstract_batteries_state = nnx.split(network_batteries_shape)

    with open(path + '/state_batteries.pkl', 'rb') as file:
        state_batteries_restored = pickle.load(file)

    network_batteries = nnx.merge(graphdef_batteries, state_batteries_restored)

    network_rec_shape = construct_rec_net_from_config_multi_agent(config, nnx.Rngs(0))
    graphdef_rec, abstract_rec_state = nnx.split(network_rec_shape)

    with open(path + '/state_rec.pkl', 'rb') as file:
        state_rec_restored = pickle.load(file)

    network_rec = nnx.merge(graphdef_rec, state_rec_restored)

    with open(path + '/val_info.pkl', 'rb') as file:
        val_info = pickle.load(file)

    return network_batteries, network_rec, config, params, val_info