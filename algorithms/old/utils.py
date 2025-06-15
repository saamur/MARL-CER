import os
from copy import deepcopy
from datetime import datetime

import numpy as np

import jax.tree
from flax.core import unfreeze

import flax.nnx as nnx
import pickle
import lzma

from algorithms.networks import (ActorCritic, RecurrentActorCritic, StackedActorCritic,
                                              StackedRecurrentActorCritic,
                                              RECActorCritic, RECRecurrentActorCritic, RECActorCriticConcat,
                                              AsymmetricRECActorCriticConcat)

from algorithms.networks_prova import StackedActorCritic as StackedActorCriticVecRECValue
from algorithms.networks_prova import RECActorCritic as RECActorCriticVecRECValue

from algorithms.networks_prova_double import StackedActorCritic as StackedActorCriticVecRECValueBoth
from algorithms.networks_prova_double import RECActorCritic as RECActorCriticVecRECValueBoth

from algorithms.networks_mappo import StackedActorCritic as StackedActorCriticMAPPO
from algorithms.networks_mappo import RECActorCritic as RECActorCriticMAPPO

from algorithms.networks_lio import StackedIncentiveNetwork, StackedIncentiveNetworkPercentage

from algorithms.networks_mpl import RECMLP

path_base = '/trained_agents/'


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
        return ActorCritic(
            config['OBSERVATION_SPACE_SIZE'],
            config['ACTION_SPACE_SIZE'],
            activation=config['ACTIVATION'],
            net_arch=config.get('NET_ARCH'),
            act_net_arch=config.get('ACT_NET_ARCH'),
            cri_net_arch=config.get('CRI_NET_ARCH'),
            add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
            rngs=rng
        )
    elif config['NETWORK'] == 'recurrent_actor_critic':
        return RecurrentActorCritic(
            config['OBSERVATION_SPACE_SIZE'],
            config['ACTION_SPACE_SIZE'],
            num_sequences=config['NUM_SEQUENCES'],
            activation=config['ACTIVATION'],
            lstm_activation=config['LSTM_ACTIVATION'],
            net_arch=config.get('NET_ARCH'),
            act_net_arch=config.get('ACT_NET_ARCH'),
            cri_net_arch=config.get('CRI_NET_ARCH'),
            lstm_net_arch=config.get('LSTM_NET_ARCH'),
            lstm_act_net_arch=config.get('LSTM_ACT_NET_ARCH'),
            lstm_cri_net_arch=config.get('LSTM_CRI_NET_ARCH'),
            add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
            rngs=rng
        )
    else:
        raise ValueError('Invalid network name')

# def construct_battery_net_from_config_multi_agent_only_actor_critic(config, rng):
#
#     @nnx.split_rngs(splits=config['NUM_BATTERY_AGENTS'])
#     def thing(rng):
#         return StackedActorCritic(
#             config['BATTERY_OBSERVATION_SPACE_SIZE'],
#             config['BATTERY_ACTION_SPACE_SIZE'],
#             activation=config['ACTIVATION'],
#             net_arch=config.get('NET_ARCH'),
#             act_net_arch=config.get('ACT_NET_ARCH'),
#             cri_net_arch=config.get('CRI_NET_ARCH'),
#             add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
#             rngs=rng)
#     return thing(rng)

def construct_battery_net_from_config_multi_agent(config, rng, num_nets=None):

    if num_nets is None:
        num_nets = config['NUM_BATTERY_AGENTS']
    # if obs_space_size is None:
    #     obs_space_size = config.get('BATTERY_OBSERVATION_SPACE_SIZE', config['BATTERY_OBSERVATION_SPACE_SIZE_ACT'])

    if config['NETWORK_TYPE_BATTERIES'] == 'actor_critic':
        return StackedActorCritic(
            num_nets,
            config['BATTERY_OBS_KEYS'],
            config['BATTERY_ACTION_SPACE_SIZE'],
            activation=config['ACTIVATION'],
            obs_keys_cri=config.get('BATTERY_OBS_KEYS_CRI', None),
            net_arch=(config['NET_ARCH_BATTERIES'] if 'NET_ARCH_BATTERIES' in config.keys() else config.get('NET_ARCH')),
            act_net_arch=(config['ACT_NET_ARCH_BATTERIES'] if 'ACT_NET_ARCH_BATTERIES' in config.keys() else config.get('ACT_NET_ARCH')),
            cri_net_arch=(config['CRI_NET_ARCH_BATTERIES'] if 'CRI_NET_ARCH_BATTERIES' in config.keys() else config.get('CRI_NET_ARCH')),
            add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
            normalize=config['NORMALIZE_NN_INPUTS'],
            # is_feature_normalizable=config['BATTERY_OBS_IS_NORMALIZABLE'],
            rngs=rng)
    # elif config['NETWORK_TYPE_BATTERIES'] == 'asymmetric_actor_critic':
    #     return StackedAsymmetricActorCritic(
    #         num_nets,
    #         obs_space_size,
    #         config['BATTERY_OBSERVATION_SPACE_SIZE_ONLY_CRI'],
    #         config['BATTERY_ACTION_SPACE_SIZE'],
    #         activation=config['ACTIVATION'],
    #         net_arch=(config['NET_ARCH_BATTERIES'] if 'NET_ARCH_BATTERIES' in config.keys() else config.get('NET_ARCH')),
    #         act_net_arch=(config['ACT_NET_ARCH_BATTERIES'] if 'ACT_NET_ARCH_BATTERIES' in config.keys() else config.get('ACT_NET_ARCH')),
    #         cri_net_arch=(config['CRI_NET_ARCH_BATTERIES'] if 'CRI_NET_ARCH_BATTERIES' in config.keys() else config.get('CRI_NET_ARCH')),
    #         add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
    #         normalize=config['NORMALIZE_NN_INPUTS'],
    #         # is_feature_normalizable=config['BATTERY_OBS_IS_NORMALIZABLE'],
    #         rngs=rng)
    elif config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
        return StackedRecurrentActorCritic(
            num_nets,
            config['BATTERY_OBS_KEYS'],
            config['BATTERY_ACTION_SPACE_SIZE'],
            activation=config['ACTIVATION'],
            obs_is_seq=config['BATTERY_OBS_IS_SEQUENCE'],
            obs_keys_cri=config.get('BATTERY_OBS_KEYS_CRI', None),
            lstm_activation=config['LSTM_ACTIVATION'],
            net_arch=(config['NET_ARCH_BATTERIES'] if 'NET_ARCH_BATTERIES' in config.keys() else config.get('NET_ARCH')),
            act_net_arch=(config['ACT_NET_ARCH_BATTERIES'] if 'ACT_NET_ARCH_BATTERIES' in config.keys() else config.get('ACT_NET_ARCH')),
            cri_net_arch=(config['CRI_NET_ARCH_BATTERIES'] if 'CRI_NET_ARCH_BATTERIES' in config.keys() else config.get('CRI_NET_ARCH')),
            lstm_net_arch=(config['LSTM_NET_ARCH_BATTERIES'] if 'LSTM_NET_ARCH_BATTERIES' in config.keys() else config.get('LSTM_NET_ARCH')),
            lstm_act_net_arch=(config['LSTM_ACT_NET_ARCH_BATTERIES'] if 'LSTM_ACT_NET_ARCH_BATTERIES' in config.keys() else config.get('LSTM_ACT_NET_ARCH')),
            lstm_cri_net_arch=(config['LSTM_CRI_NET_ARCH_BATTERIES'] if 'LSTM_CRI_NET_ARCH_BATTERIES' in config.keys() else config.get('LSTM_CRI_NET_ARCH')),
            add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
            normalize=config['NORMALIZE_NN_INPUTS'],
            # is_feature_normalizable=config['BATTERY_OBS_IS_NORMALIZABLE'],
            rngs=rng
    )
    elif config['NETWORK_TYPE_BATTERIES'] == 'actor_critic_vec':
        return StackedActorCriticVecRECValue(
            num_nets,
            config['BATTERY_OBS_KEYS'],
            config['BATTERY_ACTION_SPACE_SIZE'],
            activation=config['ACTIVATION'],
            obs_keys_cri=config.get('BATTERY_OBS_KEYS_CRI', None),
            net_arch=(config['NET_ARCH_BATTERIES'] if 'NET_ARCH_BATTERIES' in config.keys() else config.get('NET_ARCH')),
            act_net_arch=(config['ACT_NET_ARCH_BATTERIES'] if 'ACT_NET_ARCH_BATTERIES' in config.keys() else config.get('ACT_NET_ARCH')),
            cri_net_arch=(config['CRI_NET_ARCH_BATTERIES'] if 'CRI_NET_ARCH_BATTERIES' in config.keys() else config.get('CRI_NET_ARCH')),
            add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
            normalize=config['NORMALIZE_NN_INPUTS'],
            len_rec_val=config['LEN_REC_VAL'],
            # is_feature_normalizable=config['BATTERY_OBS_IS_NORMALIZABLE'],
            rngs=rng)
    elif config['NETWORK_TYPE_BATTERIES'] == 'actor_critic_vec_both':
        return StackedActorCriticVecRECValueBoth(
            num_nets,
            config['BATTERY_OBS_KEYS'],
            config['BATTERY_ACTION_SPACE_SIZE'],
            activation=config['ACTIVATION'],
            obs_keys_cri=config.get('BATTERY_OBS_KEYS_CRI', None),
            net_arch=(config['NET_ARCH_BATTERIES'] if 'NET_ARCH_BATTERIES' in config.keys() else config.get('NET_ARCH')),
            act_net_arch=(config['ACT_NET_ARCH_BATTERIES'] if 'ACT_NET_ARCH_BATTERIES' in config.keys() else config.get('ACT_NET_ARCH')),
            cri_net_arch=(config['CRI_NET_ARCH_BATTERIES'] if 'CRI_NET_ARCH_BATTERIES' in config.keys() else config.get('CRI_NET_ARCH')),
            add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
            normalize=config['NORMALIZE_NN_INPUTS'],
            len_rec_val=config['LEN_REC_VAL'],
            len_battery_val=config['LEN_BATTERY_VAL'],
            cri_finalize_net=config['CRI_FINALIZE_NET_BATTERY'],
            # is_feature_normalizable=config['BATTERY_OBS_IS_NORMALIZABLE'],
            rngs=rng)
    elif config['NETWORK_TYPE_BATTERIES'] == 'actor_critic_mappo':
        return StackedActorCriticMAPPO(
            num_nets,
            config['BATTERY_OBS_KEYS'],
            config['BATTERY_ACTION_SPACE_SIZE'],
            activation=config['ACTIVATION'],
            obs_keys_cri=config.get('BATTERY_OBS_KEYS_CRI', None),
            net_arch=(config['NET_ARCH_BATTERIES'] if 'NET_ARCH_BATTERIES' in config.keys() else config.get('NET_ARCH')),
            act_net_arch=(config['ACT_NET_ARCH_BATTERIES'] if 'ACT_NET_ARCH_BATTERIES' in config.keys() else config.get('ACT_NET_ARCH')),
            cri_net_arch=(config['CRI_NET_ARCH_BATTERIES'] if 'CRI_NET_ARCH_BATTERIES' in config.keys() else config.get('CRI_NET_ARCH')),
            add_logistic_to_actor=config['LOGISTIC_FUNCTION_TO_ACTOR'],
            normalize=config['NORMALIZE_NN_INPUTS'],
            num_batteries=config['NUM_BATTERY_AGENTS'],
            # is_feature_normalizable=config['BATTERY_OBS_IS_NORMALIZABLE'],
            rngs=rng)
    else:
        raise ValueError('Invalid network name')

def construct_rec_net_from_config_multi_agent_only_actor_critic(config, rng):
    return RECActorCritic(config['REC_INPUT_NETWORK_SIZE'],
                          config['NUM_BATTERY_AGENTS'],
                          config['ACTIVATION'],
                          rngs=rng,
                          net_arch=config.get('NET_ARCH'),
                          act_net_arch=config.get('ACT_NET_ARCH'),
                          cri_net_arch=config.get('CRI_NET_ARCH'),
                          passive_houses=config['PASSIVE_HOUSES'])

def construct_rec_net_from_config_multi_agent(config, rng):
    if config['NETWORK_TYPE_REC'] == 'actor_critic':
        return RECActorCritic(config['REC_OBS_KEYS'],
                              config['REC_OBS_IS_LOCAL'],
                              config['NUM_BATTERY_AGENTS'],
                              config['ACTIVATION'],
                              rngs=rng,
                              obs_keys_cri=config.get('REC_OBS_KEYS_CRI', None),
                              net_arch=(config['NET_ARCH_REC'] if 'NET_ARCH_REC' in config.keys() else config.get('NET_ARCH', ())),
                              act_net_arch=(config['ACT_NET_ARCH_REC'] if 'ACT_NET_ARCH_REC' in config.keys() else config.get('ACT_NET_ARCH')),
                              cri_net_arch=(config['CRI_NET_ARCH_REC'] if 'CRI_NET_ARCH_REC' in config.keys() else config.get('CRI_NET_ARCH')),
                              passive_houses=config['PASSIVE_HOUSES'],
                              normalize=config['NORMALIZE_NN_INPUTS'],
                              non_shared_net_arch_before=config.get('NON_SHARED_NET_ARCH_BEFORE', ()),
                              non_shared_net_arch_after=config.get('NON_SHARED_NET_ARCH_AFTER', ()),
                              # is_obs_normalizable=config['REC_OBS_IS_NORMALIZABLE']
                              )
    elif config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
        return RECRecurrentActorCritic(config['REC_OBS_KEYS'],
                                       config['REC_OBS_IS_LOCAL'],
                                       config['REC_OBS_IS_SEQUENCE'],
                                       config['NUM_BATTERY_AGENTS'],
                                       config['ACTIVATION'],
                                       rngs=rng,
                                       net_arch=(config['NET_ARCH_REC'] if 'NET_ARCH_REC' in config.keys() else config.get('NET_ARCH', ())),
                                       act_net_arch=(config['ACT_NET_ARCH_REC'] if 'ACT_NET_ARCH_REC' in config.keys() else config.get('ACT_NET_ARCH')),
                                       cri_net_arch=(config['CRI_NET_ARCH_REC'] if 'CRI_NET_ARCH_REC' in config.keys() else config.get('CRI_NET_ARCH')),
                                       lstm_net_arch=(config['LSTM_NET_ARCH_REC'] if 'LSTM_NET_ARCH_REC' in config.keys() else config.get('LSTM_NET_ARCH')),
                                       lstm_act_net_arch=(config['LSTM_ACT_NET_ARCH_REC'] if 'LSTM_ACT_NET_ARCH_REC' in config.keys() else config.get('LSTM_ACT_NET_ARCH')),
                                       lstm_cri_net_arch=(config['LSTM_CRI_NET_ARCH_REC'] if 'LSTM_CRI_NET_ARCH_REC' in config.keys() else config.get('LSTM_CRI_NET_ARCH')),
                                       lstm_activation=config['LSTM_ACTIVATION'],
                                       non_shared_net_arch_before=config.get('NON_SHARED_NET_ARCH_BEFORE', ()),
                                       non_shared_net_arch_after=config.get('NON_SHARED_NET_ARCH_AFTER', ()),
                                       share_lstm_batteries=config['SHARE_LSTM_BATTERIES'],
                                       passive_houses=config['PASSIVE_HOUSES'],
                                       normalize=config['NORMALIZE_NN_INPUTS'],
                                       # is_obs_normalizable=config['REC_OBS_IS_NORMALIZABLE']
                                       )
    elif config['NETWORK_TYPE_REC'] == 'actor_critic_vec':
        return RECActorCriticVecRECValue(config['REC_OBS_KEYS'],
                              config['REC_OBS_IS_LOCAL'],
                              config['NUM_BATTERY_AGENTS'],
                              config['ACTIVATION'],
                              rngs=rng,
                              obs_keys_cri=config.get('REC_OBS_KEYS_CRI', None),
                              net_arch=(config['NET_ARCH_REC'] if 'NET_ARCH_REC' in config.keys() else config.get('NET_ARCH', ())),
                              act_net_arch=(config['ACT_NET_ARCH_REC'] if 'ACT_NET_ARCH_REC' in config.keys() else config.get('ACT_NET_ARCH')),
                              cri_net_arch=(config['CRI_NET_ARCH_REC'] if 'CRI_NET_ARCH_REC' in config.keys() else config.get('CRI_NET_ARCH')),
                              passive_houses=config['PASSIVE_HOUSES'],
                              normalize=config['NORMALIZE_NN_INPUTS'],
                              non_shared_net_arch_before=config.get('NON_SHARED_NET_ARCH_BEFORE', ()),
                              non_shared_net_arch_after=config.get('NON_SHARED_NET_ARCH_AFTER', ()),
                              cri_finalize_net=config['CRI_FINALIZE_NET'],
                              len_separate_cri=config['LEN_REC_VAL'],
                              # is_obs_normalizable=config['REC_OBS_IS_NORMALIZABLE']
                              )
    elif config['NETWORK_TYPE_REC'] == 'actor_critic_vec_both':
        return RECActorCriticVecRECValueBoth(config['REC_OBS_KEYS'],
                              config['REC_OBS_IS_LOCAL'],
                              config['NUM_BATTERY_AGENTS'],
                              config['ACTIVATION'],
                              rngs=rng,
                              obs_keys_cri=config.get('REC_OBS_KEYS_CRI', None),
                              net_arch=(config['NET_ARCH_REC'] if 'NET_ARCH_REC' in config.keys() else config.get('NET_ARCH', ())),
                              act_net_arch=(config['ACT_NET_ARCH_REC'] if 'ACT_NET_ARCH_REC' in config.keys() else config.get('ACT_NET_ARCH')),
                              cri_net_arch=(config['CRI_NET_ARCH_REC'] if 'CRI_NET_ARCH_REC' in config.keys() else config.get('CRI_NET_ARCH')),
                              passive_houses=config['PASSIVE_HOUSES'],
                              normalize=config['NORMALIZE_NN_INPUTS'],
                              non_shared_net_arch_before=config.get('NON_SHARED_NET_ARCH_BEFORE', ()),
                              non_shared_net_arch_after=config.get('NON_SHARED_NET_ARCH_AFTER', ()),
                              cri_finalize_net=config['CRI_FINALIZE_NET_REC'],
                              len_separate_cri=config['LEN_REC_VAL'],
                              len_battery_val=config['LEN_BATTERY_VAL'],
                              # is_obs_normalizable=config['REC_OBS_IS_NORMALIZABLE']
                              )
    elif config['NETWORK_TYPE_REC'] == 'actor_critic_mappo':
        return RECActorCriticMAPPO(config['REC_OBS_KEYS'],
                              config['REC_OBS_IS_LOCAL'],
                              config['NUM_BATTERY_AGENTS'],
                              config['ACTIVATION'],
                              rngs=rng,
                              obs_keys_cri=config.get('REC_OBS_KEYS_CRI', None),
                              net_arch=(config['NET_ARCH_REC'] if 'NET_ARCH_REC' in config.keys() else config.get(
                                  'NET_ARCH', ())),
                              act_net_arch=(
                                  config['ACT_NET_ARCH_REC'] if 'ACT_NET_ARCH_REC' in config.keys() else config.get(
                                      'ACT_NET_ARCH')),
                              cri_net_arch=(
                                  config['CRI_NET_ARCH_REC'] if 'CRI_NET_ARCH_REC' in config.keys() else config.get(
                                      'CRI_NET_ARCH')),
                              passive_houses=config['PASSIVE_HOUSES'],
                              normalize=config['NORMALIZE_NN_INPUTS'],
                              non_shared_net_arch_before=config.get('NON_SHARED_NET_ARCH_BEFORE', ()),
                              non_shared_net_arch_after=config.get('NON_SHARED_NET_ARCH_AFTER', ()),
                              # is_obs_normalizable=config['REC_OBS_IS_NORMALIZABLE']
                              )
    elif config['NETWORK_TYPE_REC'] == 'mlp':
        return RECMLP(config['REC_OBS_KEYS'],
                      config['REC_OBS_IS_LOCAL'],
                      config['NUM_BATTERY_AGENTS'],
                      config['ACTIVATION'],
                      rngs=rng,
                      net_arch=config.get('NET_ARCH_REC', ()),
                      passive_houses=config['PASSIVE_HOUSES'],
                      normalize=config['NORMALIZE_NN_INPUTS'],
                      non_shared_net_arch_before=config.get('NON_SHARED_NET_ARCH_BEFORE', ()),
                      non_shared_net_arch_after=config.get('NON_SHARED_NET_ARCH_AFTER', ()),
                )
    else:
        raise ValueError('Invalid network name')

def construct_incentive_net_from_config_multi_agent(config, rng):
    if config.get('PERCENTAGE_INCENTIVES', False):
        return StackedIncentiveNetworkPercentage(config['NUM_RL_AGENTS'],
                                                 config['BATTERY_OBS_KEYS'],
                                                 config['NUM_BATTERY_AGENTS'],
                                                 config['ACTIVATION_INCENTIVE'],
                                                 config['INCENTIVE_NET_ARCH'],
                                                 rng,
                                                 normalize=config['NORMALIZE_NN_INPUTS'])
    else:
        return StackedIncentiveNetwork(config['NUM_RL_AGENTS'],
                                       config['BATTERY_OBS_KEYS'],
                                       config['NUM_BATTERY_AGENTS'],
                                       config['ACTIVATION_INCENTIVE'],
                                       config['INCENTIVE_NET_ARCH'],
                                       rng,
                                       normalize=config['NORMALIZE_NN_INPUTS'])


def save_state(network, config, params: dict, val_info:dict=None, train_info:dict=None, env_type='normal', additional_info=''):
    dir_name = (datetime.now().strftime('%Y%m%d_%H%M%S') +
                '_lr_' + str(config.get('LR')) +
                '_tot_timesteps_' + str(config.get('TOTAL_TIMESTEPS')) +
                '_rl_sched_' + str(config.get('LR_SCHEDULE')) +
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

    # val_info = jax.tree.map(lambda x: np.array(x), val_info)

    with open(path_base + dir_name + '/val_info.pkl', 'wb') as file:
        pickle.dump(val_info, file)

    # train_info = jax.tree.map(lambda x: np.array(x), train_info)

    with open(path_base + dir_name + '/train_info.pkl', 'wb') as file:
        pickle.dump(train_info, file)


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

    if os.path.isfile(path + '/train_info.pkl'):
        with open(path + '/train_info.pkl', 'rb') as file:
            train_info = pickle.load(file)
    else:
        train_info = None

    return network, config, params, train_info, val_info

def save_state_multiagent(directory, networks_batteries, network_rec, config: dict, world_metadata, train_info:dict=None, val_info:dict=None, is_checkpoint=False, num_steps=-1, additional_info=''):
    # dir_name = (datetime.now().strftime('%Y%m%d_%H%M%S') +
    #             '_bat_net_type_' + str(config['NETWORK_TYPE_BATTERIES']) +
    #             '_rec_net_type_' + str(config['NETWORK_TYPE_REC']) +
    #             '_lr_bat_' + str(config.get('LR_BATTERIES')) +
    #             '_lr_REC_' + str(config.get('LR_SCHEDULE')) +
    #             '_tot_timesteps_' + str(config.get('TOTAL_TIMESTEPS')) +
    #             '_lr_sched_' + str(config.get('LR_SCHEDULE')) +
    #             '_' + env_type +
    #             '_multiagent')

    if is_checkpoint:
        directory = directory + 'checkpoints/' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(num_steps) +  '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    _, state_batteries = nnx.split(networks_batteries)
    _, state_rec = nnx.split(network_rec)

    with lzma.open(directory + 'state_batteries.xz', 'wb') as file:
        pickle.dump(state_batteries, file)
    with lzma.open(directory + 'state_rec.xz', 'wb') as file:
        pickle.dump(state_rec, file)


    with lzma.open(directory + 'config.xz', 'wb') as file:
        pickle.dump(config, file)

    with lzma.open(directory + 'world_metadata.xz', 'wb') as file:
        pickle.dump(world_metadata, file)

    val_info = jax.tree.map(lambda x: np.array(x), val_info)

    with lzma.open(directory + 'val_info.xz', 'wb', preset=4) as file:
        pickle.dump(val_info, file)

    train_info = jax.tree.map(lambda x: np.array(x), train_info)

    with lzma.open(directory + 'train_info.xz', 'wb', preset=4) as file:
        pickle.dump(train_info, file)


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

    with lzma.open(path + '/world_metadata.xz', 'rb') as file:
        world_metadata = pickle.load(file)

    network_batteries_shape = construct_battery_net_from_config_multi_agent(config, nnx.Rngs(0), num_nets=config.get('NUM_RL_AGENTS'))
    graphdef_batteries, abstract_batteries_state = nnx.split(network_batteries_shape)

    with lzma.open(path + '/state_batteries.xz', 'rb') as file:
        state_batteries_restored = pickle.load(file)

    network_batteries = nnx.merge(graphdef_batteries, state_batteries_restored)

    if config.get('USE_REC_RULE_BASED_POLICY', False):
        network_rec = None
    else:
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

    return network_batteries, network_rec, config, world_metadata, train_info, val_info



def restore_state_multi_agent_adaptable_rb(path):

    with lzma.open(path + '/config.xz', 'rb') as file:
        config = pickle.load(file)

    config = unfreeze(config)

    if 'NORMALIZE_NN_INPUTS' not in config.keys():
        config['NORMALIZE_NN_INPUTS'] = False

    if 'NETWORK_TYPE_REC' not in config.keys():
        config['NETWORK_TYPE_REC'] = 'actor_critic'

    with lzma.open(path + '/world_metadata.xz', 'rb') as file:
        world_metadata = pickle.load(file)

    if config.get('USE_REC_RULE_BASED_POLICY', False):
        network_rec = None
    else:
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

    return network_rec, config, world_metadata, train_info, val_info


def save_state_multiagent_with_double_battery_net(directory, networks_batteries, network_batteries_only_local, network_rec,
                                                  config: dict, world_metadata, train_info:dict=None, val_info:dict=None,
                                                  is_checkpoint=False, num_steps=-1, additional_info=''):
    # dir_name = (datetime.now().strftime('%Y%m%d_%H%M%S') +
    #             '_bat_net_type_' + str(config['NETWORK_TYPE_BATTERIES']) +
    #             '_rec_net_type_' + str(config['NETWORK_TYPE_REC']) +
    #             '_lr_bat_' + str(config.get('LR_BATTERIES')) +
    #             '_lr_REC_' + str(config.get('LR_SCHEDULE')) +
    #             '_tot_timesteps_' + str(config.get('TOTAL_TIMESTEPS')) +
    #             '_lr_sched_' + str(config.get('LR_SCHEDULE')) +
    #             '_' + env_type +
    #             '_multiagent')

    if is_checkpoint:
        directory = directory + 'checkpoints/' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(num_steps) +  '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    _, state_batteries = nnx.split(networks_batteries)
    _, state_batteries_only_local = nnx.split(network_batteries_only_local)
    _, state_rec = nnx.split(network_rec)

    with lzma.open(directory + 'state_batteries.xz', 'wb') as file:
        pickle.dump(state_batteries, file)
    with lzma.open(directory + 'state_batteries_only_local.xz', 'wb') as file:
        pickle.dump(state_batteries_only_local, file)
    with lzma.open(directory + 'state_rec.xz', 'wb') as file:
        pickle.dump(state_rec, file)


    with lzma.open(directory + 'config.xz', 'wb') as file:
        pickle.dump(config, file)

    with lzma.open(directory + 'world_metadata.xz', 'wb') as file:
        pickle.dump(world_metadata, file)

    val_info = jax.tree.map(lambda x: np.array(x), val_info)

    with lzma.open(directory + 'val_info.xz', 'wb', preset=4) as file:
        pickle.dump(val_info, file)

    train_info = jax.tree.map(lambda x: np.array(x), train_info)

    with lzma.open(directory + 'train_info.xz', 'wb', preset=4) as file:
        pickle.dump(train_info, file)

def restore_state_multi_agent_with_double_battery_net(path):

    with lzma.open(path + '/config.xz', 'rb') as file:
        config = pickle.load(file)

    config = unfreeze(config)

    if 'NORMALIZE_NN_INPUTS' not in config.keys():
        config['NORMALIZE_NN_INPUTS'] = False

    if 'NETWORK_TYPE_BATTERIES' not in config.keys():
        config['NETWORK_TYPE_BATTERIES'] = 'actor_critic'
    if 'NETWORK_TYPE_REC' not in config.keys():
        config['NETWORK_TYPE_REC'] = 'actor_critic'

    with lzma.open(path + '/world_metadata.xz', 'rb') as file:
        world_metadata = pickle.load(file)

    network_batteries_shape = construct_battery_net_from_config_multi_agent(config, nnx.Rngs(0), num_nets=config.get('NUM_RL_AGENTS'))
    graphdef_batteries, abstract_batteries_state = nnx.split(network_batteries_shape)

    with lzma.open(path + '/state_batteries.xz', 'rb') as file:
        state_batteries_restored = pickle.load(file)

    network_batteries = nnx.merge(graphdef_batteries, state_batteries_restored)

    network_batteries_only_local_shape = construct_battery_net_from_config_multi_agent(config, nnx.Rngs(0), num_nets=config.get('NUM_RL_AGENTS'), obs_space_size=len('BATTERY_NUM_OBS_ONLY_LOCAL'))
    graphdef_batteries_only_local, abstract_batteries_only_local_state = nnx.split(network_batteries_only_local_shape)

    with lzma.open(path + '/state_batteries_only_local.xz', 'rb') as file:
        state_batteries_only_local_restored = pickle.load(file)

    network_batteries_only_local = nnx.merge(graphdef_batteries_only_local, state_batteries_only_local_restored)


    if config.get('USE_REC_RULE_BASED_POLICY', False):
        network_rec = None
    else:
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

    return network_batteries, network_batteries_only_local, network_rec, config, world_metadata, train_info, val_info




def save_state_multiagent_only_batteries(directory, networks_policy, networks_incentives, config: dict, world_metadata, train_info:dict=None, val_info:dict=None, is_checkpoint=False, num_steps=-1, additional_info=''):

    if is_checkpoint:
        directory = directory + 'checkpoints/' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(num_steps) +  '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    _, state_policy = nnx.split(networks_policy)
    _, state_incentives = nnx.split(networks_incentives)

    with lzma.open(directory + 'state_policy.xz', 'wb') as file:
        pickle.dump(state_policy, file)
    with lzma.open(directory + 'state_incentives.xz', 'wb') as file:
        pickle.dump(state_incentives, file)


    with lzma.open(directory + 'config.xz', 'wb') as file:
        pickle.dump(config, file)

    with lzma.open(directory + 'world_metadata.xz', 'wb') as file:
        pickle.dump(world_metadata, file)

    val_info = jax.tree.map(lambda x: np.array(x), val_info)

    with lzma.open(directory + 'val_info.xz', 'wb', preset=4) as file:
        pickle.dump(val_info, file)

    train_info = jax.tree.map(lambda x: np.array(x), train_info)

    with lzma.open(directory + 'train_info.xz', 'wb', preset=4) as file:
        pickle.dump(train_info, file)


def restore_state_multi_agent_only_batteries(path):

    with lzma.open(path + '/config.xz', 'rb') as file:
        config = pickle.load(file)

    config = unfreeze(config)

    print(config)

    if 'NORMALIZE_NN_INPUTS' not in config.keys():
        config['NORMALIZE_NN_INPUTS'] = False

    if 'NETWORK_TYPE_BATTERIES' not in config.keys():
        config['NETWORK_TYPE_BATTERIES'] = 'actor_critic'
    if 'NETWORK_TYPE_REC' not in config.keys():
        config['NETWORK_TYPE_REC'] = 'actor_critic'

    with lzma.open(path + '/world_metadata.xz', 'rb') as file:
        world_metadata = pickle.load(file)

    network_policy_shape = construct_battery_net_from_config_multi_agent(config, nnx.Rngs(0), num_nets=config.get('NUM_RL_AGENTS'))
    graphdef_policy, abstract_policy_state = nnx.split(network_policy_shape)

    with lzma.open(path + '/state_policy.xz', 'rb') as file:
        state_policy_restored = pickle.load(file)

    network_policy = nnx.merge(graphdef_policy, state_policy_restored)


    network_incentives_shape = construct_incentive_net_from_config_multi_agent(config, nnx.Rngs(0))
    graphdef_incentives, abstract_incentives_state = nnx.split(network_incentives_shape)

    with lzma.open(path + '/state_incentives.xz', 'rb') as file:
        state_incentives_restored = pickle.load(file)

    network_incentives = nnx.merge(graphdef_incentives, state_incentives_restored)


    if os.path.isfile(path + '/train_info.xz'):
        with lzma.open(path + '/train_info.xz', 'rb') as file:
            train_info = pickle.load(file)
    else:
        train_info = None

    with lzma.open(path + '/val_info.xz', 'rb') as file:
        val_info = pickle.load(file)

    return network_policy, network_incentives, config, world_metadata, train_info, val_info
