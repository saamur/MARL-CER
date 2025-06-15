import jax
import jax.numpy as jnp
from flax import nnx

from algorithms.train_core import RunnerState, UpdateState, Transition


def update_batteries_network(runner_state: RunnerState, traj_batch, advantages, targets, num_minibatches, minibatch_size, num_epochs, config):
    def _update_epoch(update_state: UpdateState):
        def _update_minbatch(net_and_optim, traj_batch, advantages, targets):
            network_batteries, optimizer_batteries = net_and_optim

            if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
                grad_fn_batteries = nnx.value_and_grad(ppo_loss_recurrent, has_aux=True)
            else:
                grad_fn_batteries = nnx.value_and_grad(ppo_loss, has_aux=True)

            total_loss_batteries, grads_batteries = grad_fn_batteries(
                network_batteries,
                traj_batch,
                advantages,
                targets,
                config
            )

            optimizer_batteries.update(grads_batteries)

            total_loss = total_loss_batteries

            return (network_batteries, optimizer_batteries), total_loss

        rng, _rng = jax.random.split(update_state.rng)

        batch = (traj_batch, advantages, targets)
        batch_size = minibatch_size * num_minibatches

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1), batch
            )
            batch = jax.tree.map(
                lambda x: x.reshape((x.shape[0],) + (-1, batch_size) + x.shape[2:]), batch
            )
            sequences = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]), batch
            )
            permutation = jax.random.permutation(_rng, num_minibatches)
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
                    x, (num_minibatches, -1) + x.shape[1:]
                ),
                shuffled_batch,
            )

        scanned_update_minibatch = nnx.scan(_update_minbatch,
                                            in_axes=((nnx.Carry, 0, 0, 0)))

        _, total_loss = scanned_update_minibatch((update_state.network, update_state.optimizer), *minibatches)

        update_state = update_state._replace(rng=rng)
        return update_state, total_loss

    update_state = UpdateState(network=runner_state.network_batteries, optimizer=runner_state.optimizer_batteries,
                               traj_batch=traj_batch, advantages=advantages, targets=targets, rng=runner_state.rng)

    scanned_update_epoch = nnx.scan(_update_epoch,
                                    in_axes=nnx.Carry,
                                    out_axes=(nnx.Carry, 0),
                                    length=num_epochs)

    update_state, loss_info = scanned_update_epoch(update_state)

    runner_state = runner_state._replace(rng=update_state.rng)

    return runner_state, loss_info


def ppo_loss(network, traj_batch, gae, targets, config):
    traj_batch_data = (traj_batch.obs_batteries,
                       traj_batch.actions_batteries,
                       traj_batch.values_batteries,
                       traj_batch.log_prob_batteries)

    traj_batch_obs, traj_batch_actions, traj_batch_values, traj_batch_log_probs = jax.tree.map(
        lambda x: x.swapaxes(0, 1), traj_batch_data)
    traj_batch_obs, traj_batch_actions = jax.tree.map(lambda x: x[:config['NUM_RL_AGENTS']],
                                                      (traj_batch_obs, traj_batch_actions))

    gae, targets = jax.tree.map(lambda x: x.swapaxes(0, 1), (gae, targets))

    # RERUN NETWORK
    pi, value = network(traj_batch_obs)
    log_prob = pi.log_prob(traj_batch_actions)

    if config['NORMALIZE_TARGETS_BATTERIES']:
        targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)

    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch_values + (
            value - traj_batch_values
    ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=1)

    assert value_loss.shape == (config['NUM_RL_AGENTS'],)

    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch_log_probs)

    if config['NORMALIZE_ADVANTAGES_BATTERIES']:
        gae = (gae - gae.mean(axis=1, keepdims=True)) / (gae.std(axis=1, keepdims=True) + 1e-8)

    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - config['CLIP_EPS'], 1.0 + config['CLIP_EPS']) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean(axis=1)
    entropy = pi.entropy()

    entropy = entropy.mean(axis=1)

    total_loss = (
            loss_actor
            + config['VF_COEF'] * value_loss
            - config['ENT_COEF_BATTERIES'] * entropy
    )

    assert total_loss.ndim == 1

    total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
    return total_loss, (value_loss, loss_actor, entropy)


def ppo_loss_recurrent(network, traj_batch: Transition, gae, targets, config):
    obs_battery_first = jax.tree.map(lambda x: x.swapaxes(0, 1)[:config['NUM_RL_AGENTS']], traj_batch.obs_batteries)

    data_for_network_act, data_for_network_cri = network.prepare_data(obs_battery_first, return_cri=True)

    def forward_pass_lstm(carry, data_act, data_cri, beginning):
        network, act_state, cri_state = carry

        act_state, cri_state = jax.tree.map(
            lambda x, y: jnp.where(beginning[(slice(None),) + (None,) * (x.ndim - 1)], x, y),
            network.get_initial_lstm_state(),
            (act_state, cri_state))
        # jnp.where(beginning, init_states, states)
        act_state, act_output = network.apply_lstm_act(data_act, act_state)
        cri_state, cri_output = network.apply_lstm_cri(data_cri, cri_state)
        return (network, act_state, cri_state), act_output, cri_output

    _, act_outputs, cri_outputs = nnx.scan(forward_pass_lstm,
                                           in_axes=(nnx.Carry, 1, 1, 0),
                                           out_axes=(nnx.Carry, 1, 1),
                                           unroll=16)((network,) + jax.tree.map(lambda x: x[0],
                                                                                (traj_batch.lstm_states_prev_batteries.act_state,
                                                                                 traj_batch.lstm_states_prev_batteries.cri_state)),
                                                      data_for_network_act, data_for_network_cri,
                                                      traj_batch.done_prev_batteries[:, :config['NUM_RL_AGENTS']])

    pi = network.apply_act_mlp(data_for_network_act, act_outputs)
    values = network.apply_cri_mlp(data_for_network_cri, cri_outputs)
    log_prob = pi.log_prob(traj_batch.actions_batteries[:, :config['NUM_RL_AGENTS']].swapaxes(0, 1))

    values_time_first = values.swapaxes(0, 1)
    log_prob_time_first = log_prob.swapaxes(0, 1)

    if config['NORMALIZE_TARGETS_BATTERIES']:
        targets = (targets - targets.mean(axis=1, keepdims=True)) / (targets.std(axis=1, keepdims=True) + 1e-8)

    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.values_batteries + (
            values_time_first - traj_batch.values_batteries
    ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
    value_losses = jnp.square(values_time_first - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(axis=0)

    assert value_loss.shape == (config['NUM_RL_AGENTS'],)

    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob_time_first - traj_batch.log_prob_batteries)

    if config['NORMALIZE_ADVANTAGES_BATTERIES']:
        gae = (gae - gae.mean(axis=0, keepdims=True)) / (gae.std(axis=0, keepdims=True) + 1e-8)

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
    loss_actor = loss_actor.mean(axis=0)  # time first
    entropy = pi.entropy()
    entropy = entropy.mean(axis=1)  # batteries first

    total_loss = (
            loss_actor
            + config['VF_COEF'] * value_loss
            - config['ENT_COEF_BATTERIES'] * entropy
    )

    assert total_loss.ndim == 1

    total_loss = total_loss.sum()  # the loss will be linearly dependent on the loss of the single batteries, with derivative = 1
    return total_loss, (value_loss, loss_actor, entropy)
