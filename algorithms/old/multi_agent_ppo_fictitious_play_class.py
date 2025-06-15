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
import distrax

# from jaxmarl.wrappers.baselines import JaxMARLWrapper
from algorithms.wrappers import VecEnvJaxMARL

import algorithms.utils as utils
from ernestogym.envs.multi_agent.env import RECEnv, EnvState
from algorithms.networks import StackedActorCritic, StackedRecurrentActorCritic, RECActorCritic, RECRecurrentActorCritic, RECActorCriticConcat

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


class PPOFictitiousPlay:

    def __init__(self,
                 env:RECEnv,
                 num_rl_agents:int,
                 num_battery_first_agents:int,
                 num_only_market_agents:int,
                 num_random_agents:int,
                 num_years_training: int,
                 lr_batteries:float,
                 lr_batteries_min:float,
                 lr_rec:float,
                 lr_rec_min:float,

                 num_consecutive_updates_batteries:int=1,
                 num_consecutive_updates_rec:int=1,

                 lr_schedule_batteries:str='cosine',
                 lr_schedule_rec:str='cosine',
                 optimizer_batteries:str='adamw',
                 optimizer_rec:str='adamw',

                 num_envs: int = 4,
                 num_steps: int = 8192,

                 fraction_dynamic_lr_batteries:float=1.,
                 fraction_dynamic_lr_rec:float=1.,
                 warmup_schedule_batteries:float=0.,
                 warmup_schedule_rec:float=0.,

                 update_epochs_batteries:int=10,
                 update_epochs_rec:int=10,
                 num_minibatches_batteries:int=32,
                 num_minibatches_rec:int=32,
                 gamma_batteries:float=0.99,
                 gamma_rec: float = 0.99,
                 gae_lambda_batteries:float=0.98,
                 gae_lambda_rec:float=0.98,
                 clip_eps_batteries:float=0.2,
                 clip_eps_rec:float=0.2,
                 vf_coeff_batteries:float=0.5,
                 vf_coeff_rec:float=0.5,
                 max_grad_norm_batteries:float=0.5,
                 max_grad_norm_rec:float=0.5,
                 ent_coeff_batteries:float=0.,
                 ent_coeff_rec:float=0.,
                 normalize_reward_for_gae_and_targets_batteries:bool=False,
                 normalize_reward_for_gae_and_targets_rec:bool=False,
                 normalize_targets_batteries:bool=False,
                 normalize_targets_rec:bool=False,
                 normalize_advantages_batteries:bool=False,
                 normalize_advantages_rec:bool=False,
                 network_type_batteries:str='actor_critic',
                 network_type_rec:str='actor_critic',
                 net_args_batteries:dict=None,
                 net_args_rec:dict=None,
                 normalize_nn_inputs=True,
                 imitation_learning_rec:bool=False,
                 fraction_imitation_learning_rec:float=None,
                 max_action_random_agents:float=2.,
                 ):

        self.env = VecEnvJaxMARL(env)

        self.num_battery_agents = env.num_battery_agents

        self.num_rl_agents = num_rl_agents
        self.num_battery_first_agents = num_battery_first_agents
        self.num_only_market_agents = num_only_market_agents
        self.num_random_agents = num_random_agents

        if self.num_battery_agents != (self.num_rl_agents + self.num_battery_first_agents + self.num_only_market_agents + self.num_random_agents):
            raise ValueError('The provided numbers of agents do not sum up to the number of agents of the environment')


        self.num_years_training = num_years_training

        # Batteries-related parameters
        self.lr_batteries = lr_batteries
        self.lr_batteries_min = lr_batteries_min
        self.num_consecutive_updates_batteries = num_consecutive_updates_batteries
        self.lr_schedule_batteries = lr_schedule_batteries
        self.optimizer_batteries = optimizer_batteries
        self.fraction_dynamic_lr_batteries = fraction_dynamic_lr_batteries
        self.warmup_schedule_batteries = warmup_schedule_batteries
        self.update_epochs_batteries = update_epochs_batteries
        self.num_minibatches_batteries = num_minibatches_batteries
        self.gamma_batteries = gamma_batteries
        self.gae_lambda_batteries = gae_lambda_batteries
        self.clip_eps_batteries = clip_eps_batteries
        self.vf_coeff_batteries = vf_coeff_batteries
        self.max_grad_norm_batteries = max_grad_norm_batteries
        self.ent_coeff_batteries = ent_coeff_batteries
        self.normalize_reward_for_gae_and_targets_batteries = normalize_reward_for_gae_and_targets_batteries
        self.normalize_targets_batteries = normalize_targets_batteries
        self.normalize_advantages_batteries = normalize_advantages_batteries
        self.network_type_batteries = network_type_batteries
        self.net_args_batteries = net_args_batteries if net_args_batteries is not None else {}

        # REC-related parameters
        self.lr_rec = lr_rec
        self.lr_rec_min = lr_rec_min
        self.num_consecutive_updates_rec = num_consecutive_updates_rec
        self.lr_schedule_rec = lr_schedule_rec
        self.optimizer_rec = optimizer_rec
        self.fraction_dynamic_lr_rec = fraction_dynamic_lr_rec
        self.warmup_schedule_rec = warmup_schedule_rec
        self.update_epochs_rec = update_epochs_rec
        self.num_minibatches_rec = num_minibatches_rec
        self.gamma_rec = gamma_rec
        self.gae_lambda_rec = gae_lambda_rec
        self.clip_eps_rec = clip_eps_rec
        self.vf_coeff_rec = vf_coeff_rec
        self.max_grad_norm_rec = max_grad_norm_rec
        self.ent_coeff_rec = ent_coeff_rec
        self.normalize_reward_for_gae_and_targets_rec = normalize_reward_for_gae_and_targets_rec
        self.normalize_targets_rec = normalize_targets_rec
        self.normalize_advantages_rec = normalize_advantages_rec
        self.network_type_rec = network_type_rec
        self.net_args_rec = net_args_rec if net_args_rec is not None else {}
        self.imitation_learning_rec = imitation_learning_rec
        self.fraction_imitation_learning_rec = fraction_imitation_learning_rec

        # Shared parameters
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.normalize_nn_inputs = normalize_nn_inputs


        self.total_timesteps = 8760 * self.num_envs * self.num_years_training
        self.num_updates = self.total_timesteps // self.num_steps // self.num_envs
        self.minibatch_size_batteries = self.num_envs * self.num_steps // self.num_minibatches_batteries
        self.minibatch_size_rec = self.num_envs * self.num_steps // self.num_minibatches_rec

        self.battery_action_space_size = env.action_space(env.battery_agents[0]).shape[0]
        self.battery_obs_keys = tuple(env.obs_battery_agents_keys)
        self.battery_obs_is_sequence = env.obs_is_sequence_battery
        self.battery_obs_is_normalizable = env.obs_is_normalizable_battery

        self.rec_action_space_size = env.action_space(env.rec_agent).shape[0]
        self.rec_obs_keys = tuple(env.obs_rec_keys)

        self.passive_houses = (env.num_passive_houses > 0)
        self.rec_obs_is_sequence = env.obs_is_sequence_rec
        self.rec_obs_is_local = env.obs_is_local_rec
        self.rec_obs_is_normalizable = env.obs_is_normalizable_rec

        self.network_batteries = self._construct_network_batteries(nnx.Rngs(123))
        self.network_rec = self._construct_network_rec(nnx.Rngs(222))

        def schedule_builder(lr_init, lr_end, frac_dynamic, num_updates, update_epochs, num_minibatches, warm_up, lr_schedule):

            tot_steps = int(num_minibatches * update_epochs * num_updates * frac_dynamic)
            warm_up_steps = int(num_minibatches * update_epochs * num_updates * warm_up)

            if lr_schedule == 'linear':
                return optax.schedules.linear_schedule(lr_init, lr_end, tot_steps)
            elif lr_schedule == 'cosine':
                optax.schedules.cosine_decay_schedule(lr_init, tot_steps, lr_end / lr_init)
                return optax.schedules.warmup_cosine_decay_schedule(0., lr_init, warm_up_steps, tot_steps, lr_end)
            else:
                return lr_init

        schedule_batteries = schedule_builder(self.lr_schedule_batteries,
                                              self.lr_batteries_min,
                                              self.fraction_dynamic_lr_batteries,
                                              self.num_updates * self.num_consecutive_updates_batteries / (self.num_consecutive_updates_batteries + self.num_consecutive_updates_rec),
                                              self.update_epochs_batteries,
                                              self.num_minibatches_batteries,
                                              self.warmup_schedule_batteries,
                                              self.lr_schedule_batteries)

        schedule_rec = schedule_builder(self.lr_schedule_rec,
                                        self.lr_rec_min,
                                        self.fraction_dynamic_lr_rec,
                                        self.num_updates * self.num_consecutive_updates_rec / (self.num_consecutive_updates_batteries + self.num_consecutive_updates_rec),
                                        self.update_epochs_rec,
                                        self.num_minibatches_rec,
                                        self.warmup_schedule_rec,
                                        self.lr_schedule_rec)

        def get_optim(name, scheduler):
            if name == 'adam':
                return optax.adam(learning_rate=scheduler, eps=1e-5)
            elif name == 'adamw':
                return optax.adamw(learning_rate=scheduler, eps=1e-5)
            elif name == 'sgd':
                return optax.sgd(learning_rate=scheduler)
            elif name == 'rmsprop':
                return optax.rmsprop(learning_rate=scheduler, momentum=0.9)
            else:
                raise ValueError("Optimizer '{}' not recognized".format(name))

        tx_bat = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm_batteries),
            get_optim(self.optimizer_batteries, schedule_batteries),
        )
        tx_rec = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm_rec),
            get_optim(self.optimizer_rec, schedule_rec)
        )

        self.optimizer_batteries = StackedOptimizer(self.num_rl_agents, self.network_batteries, tx_bat)
        self.optimizer_rec = nnx.Optimizer(self.network_rec, tx_rec)


    def _construct_network_batteries(self, rng):

        if self.network_type_batteries == 'actor_critic':
            return StackedActorCritic(
                self.num_rl_agents,
                self.battery_obs_keys,
                self.battery_action_space_size,
                **self.net_args_batteries,
                rngs=rng)
        elif self.network_type_batteries == 'recurrent_actor_critic':
            return StackedRecurrentActorCritic(
                self.num_rl_agents,
                self.battery_obs_keys,
                self.battery_action_space_size,
                obs_is_seq=self.battery_obs_is_sequence,
                **self.net_args_batteries,
                rngs=rng
            )
        else:
            raise ValueError('Invalid network name')

    def _construct_network_rec(self, rng):

        if self.network_type_rec == 'actor_critic':
            return RECActorCritic(self.rec_obs_keys,
                                  self.rec_obs_is_local,
                                  self.num_battery_agents,
                                  **self.net_args_rec,
                                  passive_houses=self.passive_houses,
                                  rngs=rng
                                  )
        elif self.network_type_rec == 'recurrent_actor_critic':
            return RECRecurrentActorCritic(self.rec_obs_keys,
                                           self.rec_obs_is_local,
                                           self.rec_obs_is_sequence,
                                           self.num_battery_agents,
                                           **self.net_args_rec,
                                           passive_houses=self.passive_houses,
                                           rngs=rng,
                                           )
        elif self.network_type_rec == 'actor_critic_concat':
            return RECActorCriticConcat(self.rec_obs_keys,
                                        self.rec_obs_is_local,
                                        self.num_battery_agents,
                                        **self.net_args_rec,
                                        passive_houses=self.passive_houses,
                                        rngs=rng,
                                        )
        else:
            raise ValueError('Invalid network name')