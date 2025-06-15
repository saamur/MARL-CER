import jax
import jax.numpy as jnp

from flax import nnx
import optax

from algorithms.train_core import StackedOptimizer, RunnerState, UpdateState, Transition
from algorithms.train_core import collect_trajectories, update_batteries_network, calculate_gae_batteries

from algorithms.rec_rule_based_policies import rec_rule_based_policy


def update_rec_network(runner_state:RunnerState, traj_batch:Transition, advantages, targets, iteration, config, aided=False):
    def _update_epoch(update_state: UpdateState, epoch):
        def _update_minbatch(net_and_optim, traj_batch, advantages, targets):
            network_rec, optimizer_rec = net_and_optim

            net_type = config['NETWORK_TYPE_REC']

            if aided:
                if net_type == 'recurrent_actor_critic':
                    loss_fn = ppo_aided_loss_recurrent_fn
                else:
                    loss_fn = ppo_aided_loss_fn
            else:
                if net_type == 'recurrent_actor_critic':
                    loss_fn = ppo_loss_recurrent_fn
                else:
                    loss_fn = ppo_loss_fn

            grad_fn_rec = nnx.value_and_grad(loss_fn, has_aux=True)

            total_loss_rec, grads_rec = grad_fn_rec(network_rec, traj_batch, advantages, targets,
                                                    iteration, epoch, config)

            optimizer_rec.update(grads_rec)

            return (network_rec, optimizer_rec), total_loss_rec

        rng, _rng = jax.random.split(update_state.rng)
        batch_size = config['MINIBATCH_SIZE_REC'] * config['NUM_MINIBATCHES_REC']
        assert (
                batch_size == config['NUM_STEPS'] * config['NUM_ENVS']
        ), 'batch size must be equal to number of steps * number of envs'

        batch = (traj_batch, advantages, targets)


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
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [config['NUM_MINIBATCHES_REC'], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

        scanned_update_minibatch = nnx.scan(_update_minbatch,
                                            in_axes=((nnx.Carry, 0, 0, 0)))

        _, total_loss = scanned_update_minibatch((update_state.network, update_state.optimizer), *minibatches)

        update_state = update_state._replace(rng=rng)
        return update_state, total_loss

    update_state = UpdateState(network=runner_state.network_rec, optimizer=runner_state.optimizer_rec,
                               traj_batch=traj_batch, advantages=advantages, targets=targets, rng=runner_state.rng)

    scanned_update_epoch = nnx.scan(_update_epoch,
                                    in_axes=(nnx.Carry, 0),
                                    out_axes=(nnx.Carry, 0))

    update_state, loss_info = scanned_update_epoch(update_state, jnp.arange(config['NUM_EPOCHS_REC']))

    runner_state = runner_state._replace(rng=update_state.rng)

    if aided:

        separate_losses = loss_info[1]
        jax.debug.print('ppo loss: val {v}, act {a}, ent {e}', v=separate_losses[0].mean(), a=separate_losses[1].mean(), e=separate_losses[2].mean(), ordered=True)
        jax.debug.print('imit loss: val {v}, act {a}', a=separate_losses[3].mean(), v=separate_losses[4].mean(), ordered=True)
        jax.debug.print('mean alpha: {a}', a=separate_losses[5].mean(), ordered=True)


        #FIXME DUBUG
        def my_thing(net, obs):
            pi, _ = net(obs)
            y = rec_rule_based_policy(obs, config['REC_RULE_BASED_NAME'], jax.random.PRNGKey(0))
            y_pred = pi.mean()
            return jnp.mean(jnp.abs(y_pred - y))

        jax.debug.print('mean distance rec rule-based iter {i}: {x}', i=iteration, x=my_thing(runner_state.network_rec, traj_batch.obs_rec), ordered=True)


    return runner_state, loss_info

def update_rec_network_lola_inspired(runner_state:RunnerState, env, config):

    def update_step(runner_state:RunnerState):

        def rec_loss(rec_net, runner_state:RunnerState):

            runner_state = runner_state._replace(network_rec=rec_net)

            env_state, last_obs_batteries = runner_state.env_state, runner_state.last_obs_batteries

            runner_state, traj_batch, last_val_batteries, _ = collect_trajectories(runner_state, config, env, config['NUM_STEPS_FOR_REC_UPDATE'])

            advantages_batteries, targets_batteries = calculate_gae_batteries(traj_batch, last_val_batteries, config)

            runner_state.network_batteries.train()
            update_batteries_network(runner_state, traj_batch, advantages_batteries, targets_batteries,
                                     config['NUM_MINIBATCHES_BATTERIES_FOR_REC_UPDATE'],
                                     config['NUM_STEPS_FOR_REC_UPDATE'] * config['NUM_ENVS'] // config['NUM_MINIBATCHES_BATTERIES_FOR_REC_UPDATE'],
                                     config['NUM_EPOCHS_BATTERIES_FOR_REC_UPDATE'], config)
            runner_state.network_batteries.eval()

            runner_state = runner_state._replace(env_state=env_state, last_obs_batteries=last_obs_batteries)

            runner_state, traj_batch, last_val_batteries, _ = collect_trajectories(runner_state, config, env,
                                                                                   config['NUM_STEPS_FOR_REC_UPDATE'],
                                                                                   deterministic_batteries=config.get('DETERMINISTIC_BATTERIES_FOR_REC_UPDATE', True))

            reward_rec = traj_batch.reward_rec

            loss = - reward_rec.mean()

            return loss, runner_state

        runner_state.network_rec.train()

        rec_net = runner_state.network_rec
        opt_rec = runner_state.optimizer_rec

        runner_state = runner_state._replace(network_rec=None, optimizer_rec=None)

        grad_fun = nnx.value_and_grad(rec_loss, has_aux=True)

        val, grad = grad_fun(rec_net, runner_state)
        loss, runner_state = val

        opt_rec.update(grad)

        runner_state = runner_state._replace(network_rec=rec_net, optimizer_rec=opt_rec)

        runner_state.network_rec.eval()

        return runner_state, loss, optax.global_norm(grad)

    battery_network_graph, battery_network_state = nnx.split(runner_state.network_batteries)
    battery_network_for_rec_update = nnx.merge(battery_network_graph, battery_network_state)

    if config.get('CHANGE_OPTIMIZER_BATTERIES_FOR_REC_UPDATE', True):
        battery_optimizer_for_rec_update = StackedOptimizer(config['NUM_BATTERY_AGENTS'],
                                                            battery_network_for_rec_update,
                                                            optax.chain(optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
                                                                        optax.sgd(learning_rate=config['LR_BATTERIES_FOR_REC_UPDATE'])))
    else:
        battery_optimizer_graph, battery_optimizer_state = nnx.split(runner_state.optimizer_batteries)
        battery_optimizer_for_rec_update = nnx.merge(battery_optimizer_graph, battery_optimizer_state)

    runner_state_for_rec_update = runner_state._replace(network_batteries=battery_network_for_rec_update,
                                                        optimizer_batteries=battery_optimizer_for_rec_update)

    scanned_update_step = nnx.scan(update_step,
                                   in_axes=nnx.Carry,
                                   out_axes=(nnx.Carry, 0, 0),
                                   length=config['UPDATE_TIMES_REC'])

    runner_state_for_rec_update, losses, grad_norms = scanned_update_step(runner_state_for_rec_update)

    runner_state = runner_state._replace(network_rec=runner_state_for_rec_update.network_rec,
                                         optimizer_rec=runner_state_for_rec_update.optimizer_rec)

    jax.debug.print('rec mean loss: {l}, rec mean grad norms: {n}', l=losses.mean(), n=grad_norms.mean(), ordered=True)

    return runner_state, losses


def ppo_loss_fn(network, traj_batch, gae, targets, iteration, epoch, config):
    traj_batch_obs = traj_batch.obs_rec
    traj_batch_actions = traj_batch.actions_rec
    traj_batch_values = traj_batch.value_rec
    traj_batch_log_probs = traj_batch.log_prob_rec

    # RERUN NETWORK
    pi, value = network(traj_batch_obs)
    log_prob = pi.log_prob(traj_batch_actions + 1e-8)

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
    return total_loss, (value_loss, loss_actor, entropy)


def ppo_loss_recurrent_fn(network, traj_batch: Transition, gae, targets, iteration, epoch, config):
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
                                           unroll=16)((network,) + jax.tree.map(lambda x: x[0],
                                                                                (traj_batch.lstm_states_prev_rec.act_state,
                                                                                 traj_batch.lstm_states_prev_rec.cri_state)),
                                                      data_for_network_act, data_for_network_cri,
                                                      traj_batch.done_prev_rec)

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

    return total_loss, (value_loss, loss_actor, entropy)


def loss_to_rule_based_fn(network, traj_batch, targets, config):
    pi, value = network(traj_batch.obs_rec)
    rule_based_actions = rec_rule_based_policy(traj_batch.obs_rec, config['REC_RULE_BASED_NAME'], jax.random.PRNGKey(0))

    if config.get('USE_NLL_FOR_IMITATION_LEARNING', True):
        act_loss = - pi.log_prob(rule_based_actions + 1e-8).mean()
    else:
        act_loss = 0.5 * ((pi.mean() - rule_based_actions) ** 2).sum(
            axis=-1).mean()
    act_loss *= config.get('ACTOR_LOSS_IMITATION_LEARNING_SCALE', 1)

    if config['NORMALIZE_TARGETS_REC']:
        targets = (targets - targets.mean()) / (targets.std() + 1e-8)
    critic_mse = 0.5 * jnp.mean((value - targets) ** 2)

    return act_loss + critic_mse, (act_loss, critic_mse)

def ppo_aided_loss_fn(network, traj_batch, gae, targets, iteration, epoch, config):
    ppo_loss, ppo_aux = ppo_loss_fn(network, traj_batch, gae, targets, iteration, epoch, config)
    rule_based_loss, rule_based_aux = loss_to_rule_based_fn(network, traj_batch, targets, config)
    
    tot_num_updates = (config['NUM_EPOCHS_REC'] * config['NUM_ITERATIONS'] * config.get('NUM_CONSECUTIVE_ITERATIONS_REC', 1)
                       / (config.get('NUM_CONSECUTIVE_ITERATIONS_BATTERIES', 0) + config.get('NUM_CONSECUTIVE_ITERATIONS_REC', 1)))
    
    alpha = optax.schedules.cosine_decay_schedule(1, tot_num_updates * config['FRACTION_IMITATION_LEARNING'])(iteration * config['NUM_EPOCHS_REC'] + epoch)

    loss = (1 - alpha) * ppo_loss + alpha * rule_based_loss
    return loss, ppo_aux + rule_based_aux + (alpha,)

def ppo_aided_loss_recurrent_fn(network, traj_batch, gae, targets, iteration, epoch, config):
    ppo_loss_recurrent, ppo_aux = ppo_loss_recurrent_fn(network, traj_batch, gae, targets, iteration, epoch, config)
    rule_based_loss, rule_based_aux = loss_to_rule_based_fn(network, traj_batch, targets, config)
    tot_num_updates = (config['NUM_EPOCHS_REC'] * config['NUM_ITERATIONS'] * config.get('NUM_CONSECUTIVE_ITERATIONS_REC', 1)
                       / (config.get('NUM_CONSECUTIVE_ITERATIONS_BATTERIES', 0) + config.get('NUM_CONSECUTIVE_ITERATIONS_REC', 1)))

    alpha = optax.schedules.cosine_decay_schedule(1, tot_num_updates * config['FRACTION_IMITATION_LEARNING'])(iteration * config['NUM_EPOCHS_REC'] + epoch)


    loss = (1 - alpha) * ppo_loss_recurrent + alpha * rule_based_loss
    return loss, ppo_aux + rule_based_aux + (alpha,)