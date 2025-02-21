from datetime import datetime
import orbax.checkpoint as ocp
import flax.nnx as nnx
import pickle

from algorithms.ppo import construct_net_from_config



path_base = '/media/samuele/Disco/PycharmProjectsUbuntu/MARL-CER/trained_agents/'

def save_state(network, config, params, env_type='normal', additional_info=''):
    dir_name = (datetime.now().strftime("%Y%m%d_%H%M%S") +
                '_lr_' + str(config['LR']) +
                '_tot_timesteps_' + str(config['TOTAL_TIMESTEPS']) +
                '_anneal_rl_' + str(config['ANNEAL_LR']) +
                '_' + env_type)

    if additional_info != '':
        dir_name += '_' + additional_info

    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    _, state = nnx.split(network)
    checkpointer.save(path_base + dir_name + '/state', state)

    with open(path_base + dir_name + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)

    with open(path_base + dir_name + '/params.pkl', 'wb') as file:
        pickle.dump(params, file)

def restore_state(path):

    with open(path + '/config.pkl', 'rb') as file:
        config = pickle.load(file)

    with open(path + '/params.pkl', 'rb') as file:
        params = pickle.load(file)

    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())

    network_shape = construct_net_from_config(config, nnx.Rngs(0))
    graphdef, abstract_state = nnx.split(network_shape)
    state_restored = checkpointer.restore(path + '/state', abstract_state)

    network = nnx.merge(graphdef, state_restored)

    return network, config, params