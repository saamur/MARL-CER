import os
from copy import deepcopy
from datetime import datetime

from flax.core import unfreeze

import flax.nnx as nnx
import pickle
import lzma

import algorithms.ppo as ppo
import algorithms.recurrent_ppo_wo_normalization as recurrent_ppo
import algorithms.multi_agent_ppo_only_actor_critic as multi_agent_ppo_only_actor_critic
import algorithms.multi_agent_ppo as multi_agent_ppo


path_base = '/media/samuele/Disco/PycharmProjectsUbuntu/MARL-CER/trained_agents/'


def activation_from_name(name: str):
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

def construct_battery_net_from_config_multi_agent_only_actor_critic(config, rng):

    @nnx.split_rngs(splits=config['NUM_BATTERY_AGENTS'])
    def thing(rng):
        return multi_agent_ppo_only_actor_critic.StackedActorCritic(
            config["BATTERY_OBSERVATION_SPACE_SIZE"],
            config["BATTERY_ACTION_SPACE_SIZE"],
            activation=config["ACTIVATION"],
            net_arch=config.get("NET_ARCH"),
            act_net_arch=config.get("ACT_NET_ARCH"),
            cri_net_arch=config.get("CRI_NET_ARCH"),
            add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
            rngs=rng)
    return thing(rng)

def construct_battery_net_from_config_multi_agent(config, rng):

    if config['NETWORK_TYPE_BATTERIES'] == 'actor_critic':
        return multi_agent_ppo.StackedActorCritic(
            config['NUM_BATTERY_AGENTS'],
            config["BATTERY_OBSERVATION_SPACE_SIZE"],
            config["BATTERY_ACTION_SPACE_SIZE"],
            activation=config["ACTIVATION"],
            net_arch=config.get("NET_ARCH"),
            act_net_arch=config.get("ACT_NET_ARCH"),
            cri_net_arch=config.get("CRI_NET_ARCH"),
            add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
            normalize=config["NORMALIZE_NN_INPUTS"],
            rngs=rng)
    elif config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
        return multi_agent_ppo.StackedRecurrentActorCritic(
            config['NUM_BATTERY_AGENTS'],
            config["BATTERY_OBSERVATION_SPACE_SIZE"],
            config["BATTERY_ACTION_SPACE_SIZE"],
            num_sequences=config["BATTERY_NUM_SEQUENCES"],
            activation=config["ACTIVATION"],
            lstm_activation=config["LSTM_ACTIVATION"],
            net_arch=config.get("NET_ARCH"),
            act_net_arch=config.get("ACT_NET_ARCH"),
            cri_net_arch=config.get("CRI_NET_ARCH"),
            lstm_net_arch=config.get("LSTM_NET_ARCH"),
            lstm_act_net_arch=config.get("LSTM_ACT_NET_ARCH"),
            lstm_cri_net_arch=config.get("LSTM_CRI_NET_ARCH"),
            add_logistic_to_actor=config["LOGISTIC_FUNCTION_TO_ACTOR"],
            normalize=config["NORMALIZE_NN_INPUTS"],
            rngs=rng
    )
    else:
        raise ValueError('Invalid network name')

def construct_rec_net_from_config_multi_agent_only_actor_critic(config, rng):
    return multi_agent_ppo_only_actor_critic.RECActorCritic(config["REC_INPUT_NETWORK_SIZE"],
                                                            config['NUM_BATTERY_AGENTS'],
                                                            config['ACTIVATION'],
                                                            rngs=rng,
                                                            net_arch=config.get("NET_ARCH"),
                                                            act_net_arch=config.get("ACT_NET_ARCH"),
                                                            cri_net_arch=config.get("CRI_NET_ARCH"),
                                                            passive_houses=config['PASSIVE_HOUSES'])

def construct_rec_net_from_config_multi_agent(config, rng):
    if config['NETWORK_TYPE_REC'] == 'actor_critic':
        return multi_agent_ppo.RECActorCritic(config['REC_OBS_KEYS'],
                                              config['REC_OBS_IS_LOCAL'],
                                              config['NUM_BATTERY_AGENTS'],
                                              config['ACTIVATION'],
                                              rngs=rng,
                                              net_arch=config.get("NET_ARCH"),
                                              act_net_arch=config.get("ACT_NET_ARCH"),
                                              cri_net_arch=config.get("CRI_NET_ARCH"),
                                              passive_houses=config['PASSIVE_HOUSES'],
                                              normalize=config["NORMALIZE_NN_INPUTS"])
    elif config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
        return multi_agent_ppo.RECRecurrentActorCritic(config['REC_OBS_KEYS'],
                                                       config['REC_OBS_IS_LOCAL'],
                                                       config['REC_OBS_IS_SEQUENCE'],
                                                       config['NUM_BATTERY_AGENTS'],
                                                       config['ACTIVATION'],
                                                       rngs=rng,
                                                       net_arch=config.get("NET_ARCH"),
                                                       act_net_arch=config.get("ACT_NET_ARCH"),
                                                       cri_net_arch=config.get("CRI_NET_ARCH"),
                                                       lstm_net_arch=config.get("LSTM_NET_ARCH"),
                                                       lstm_act_net_arch=config.get("LSTM_ACT_NET_ARCH"),
                                                       lstm_cri_net_arch=config.get("LSTM_CRI_NET_ARCH"),
                                                       lstm_activation=config["LSTM_ACTIVATION"],
                                                       passive_houses=config['PASSIVE_HOUSES'],
                                                       normalize=config["NORMALIZE_NN_INPUTS"])
    else:
        raise ValueError('Invalid network name')


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

def save_state_multiagent(networks_batteries, network_rec, config, params: dict, train_info:dict=None, val_info:dict=None, env_type='normal', additional_info=''):
    dir_name = (datetime.now().strftime("%Y%m%d_%H%M%S") +
                '_bat_net_type_' + str(config['NETWORK_TYPE_BATTERIES']) +
                '_rec_net_type_' + str(config['NETWORK_TYPE_REC']) +
                '_lr_bat_' + str(config.get('LR_BATTERIES')) +
                '_lr_REC_' + str(config.get('LR_SCHEDULE')) +
                '_tot_timesteps_' + str(config.get('TOTAL_TIMESTEPS')) +
                '_lr_sched_' + str(config.get('LR_SCHEDULE')) +
                '_' + env_type +
                '_multiagent')

    if additional_info != '':
        dir_name += '_' + additional_info

    os.makedirs(path_base + dir_name)

    _, state_batteries = nnx.split(networks_batteries)
    _, state_rec = nnx.split(network_rec)

    with lzma.open(path_base + dir_name + '/state_batteries.xz', 'wb') as file:
        pickle.dump(state_batteries, file)
    with lzma.open(path_base + dir_name + '/state_rec.xz', 'wb') as file:
        pickle.dump(state_rec, file)


    with lzma.open(path_base + dir_name + '/config.xz', 'wb') as file:
        pickle.dump(config, file)

    params = deepcopy(params)

    for record in (params['demands_battery_houses'] + params['demands_passive_houses'] +
                   params['generations_battery_houses'] + params['generations_passive_houses'] +
                   params['selling_prices_battery_houses'] + params['selling_prices_passive_houses'] +
                   params['buying_prices_battery_houses'] + params['buying_prices_passive_houses'] +
                   params['temp_amb_battery_houses']):
        if 'data' in record.keys():
            del record['data']

    with lzma.open(path_base + dir_name + '/params.xz', 'wb') as file:
        pickle.dump(params, file)

    with lzma.open(path_base + dir_name + '/train_info.xz', 'wb') as file:
        pickle.dump(train_info, file)

    with lzma.open(path_base + dir_name + '/val_info.xz', 'wb') as file:
        pickle.dump(val_info, file)


def restore_state_multi_agent(path):

    with lzma.open(path + '/config.xz', 'rb') as file:
        config = pickle.load(file)

    config = unfreeze(config)

    if 'NORMALIZE_NN_INPUTS' not in config.keys():
        config['NORMALIZE_NN_INPUTS'] = False

    if 'NETWORK_TYPE_BATTERIES' not in config.keys():
        config['NETWORK_TYPE_BATTERIES'] = 'actor_critic'
    if 'NETWORK_TYPE_REC' not in config.keys():
        config['NETWORK_TYPE_REC'] = 'actor_critic'

    with lzma.open(path + '/params.xz', 'rb') as file:
        params = pickle.load(file)

    network_batteries_shape = construct_battery_net_from_config_multi_agent(config, nnx.Rngs(0))
    graphdef_batteries, abstract_batteries_state = nnx.split(network_batteries_shape)

    with lzma.open(path + '/state_batteries.xz', 'rb') as file:
        state_batteries_restored = pickle.load(file)

    network_batteries = nnx.merge(graphdef_batteries, state_batteries_restored)

    network_rec_shape = construct_rec_net_from_config_multi_agent(config, nnx.Rngs(0))
    graphdef_rec, abstract_rec_state = nnx.split(network_rec_shape)

    with lzma.open(path + '/state_rec.xz', 'rb') as file:
        state_rec_restored = pickle.load(file)

    network_rec = nnx.merge(graphdef_rec, state_rec_restored)

    if os.path.isfile(path + '/train_info.xz'):
        with lzma.open(path + '/train_info.xz', 'rb') as file:
            train_info = pickle.load(file)
    else:
        train_info = None

    with lzma.open(path + '/val_info.xz', 'rb') as file:
        val_info = pickle.load(file)

    return network_batteries, network_rec, config, params, train_info, val_info