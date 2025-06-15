import jax
import jax.numpy as jnp

from flax import nnx
from algorithms.train_core.multi_agent_ppo_core import RunnerState, Transition, LSTMState
from algorithms.rec_rule_based_policies import rec_rule_based_policy


def collect_trajectories(runner_state:RunnerState, config, env, num_steps, deterministic_batteries=False, deterministic_rec=False):

    def _env_step(runner_state: RunnerState):

        network_batteries, network_rec = runner_state.network_batteries, runner_state.network_rec

        # BATTERY TURN
        runner_state, actions_batteries, value_batteries, log_prob_batteries = compute_battery_actions(runner_state, config, network_batteries, deterministic_batteries)

        actions_first = {env.battery_agents[i]: actions_batteries[:, i] for i in range(env.num_battery_agents)}
        actions_first[env.rec_agent] = jnp.zeros((config['NUM_ENVS'], env.num_battery_agents))

        rng, _rng = jax.random.split(runner_state.rng)
        rng_step = jax.random.split(_rng, config['NUM_ENVS'])
        env_state = runner_state.env_state
        obsv, env_state, reward_first, done_first, info_first = env.step(rng_step, env_state, actions_first)

        info_first['actions'] = actions_first

        rec_obsv = obsv[env.rec_agent]

        runner_state = runner_state._replace(rng=_rng)

        # REC TURN
        runner_state, actions_rec, value_rec, log_probs_rec = compute_rec_action(runner_state, config, rec_obsv, network_rec, deterministic_rec)

        actions_second = {agent: jnp.zeros((config['NUM_ENVS'], 1)) for agent in env.battery_agents}
        actions_second[env.rec_agent] = actions_rec

        assert actions_rec.shape == (config['NUM_ENVS'], env.num_battery_agents)

        rng, _rng = jax.random.split(runner_state.rng)
        rng_step = jax.random.split(_rng, config['NUM_ENVS'])
        obsv, env_state, reward_second, done_second, info_second = env.step(rng_step, env_state, actions_second)

        info_second['actions'] = actions_second

        # END STEP COMPUTATIONS

        done = jax.tree.map(jnp.logical_or, done_first, done_second)
        done_batteries = jnp.stack([done[a] for a in env.battery_agents], axis=1)
        done_rec = done[env.rec_agent]

        rewards_tot = jax.tree.map(lambda x, y: x + y, reward_first, reward_second)
        rewards_batteries = jnp.stack([rewards_tot[a] for a in env.battery_agents], axis=1)
        reward_rec = rewards_tot[env.rec_agent]

        info = jax.tree.map(lambda x, y: x + y, info_first, info_second)

        obs_batteries = jax.tree.map(lambda *vals: jnp.stack(vals, axis=1), *[obsv[a] for a in env.battery_agents])

        transition = Transition(
            done_batteries, done_rec,
            actions_batteries, actions_rec,
            value_batteries, value_rec,
            rewards_batteries, reward_rec,
            log_prob_batteries, log_probs_rec,
            runner_state.last_obs_batteries, rec_obsv,
            info,
            runner_state.done_prev_batteries, runner_state.done_prev_rec,
            runner_state.last_lstm_state_batteries, runner_state.last_lstm_state_rec
        )

        new_done_prev_batteries = done_batteries[:, :config['NUM_RL_AGENTS']] if (config['NUM_RL_AGENTS']>0 and config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic') else runner_state.done_prev_batteries
        new_done_prev_rec = done_rec if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic' else runner_state.done_prev_rec

        runner_state = runner_state._replace(env_state=env_state,
                                             last_obs_batteries=obs_batteries,
                                             rng=rng,
                                             done_prev_batteries=new_done_prev_batteries,
                                             done_prev_rec=new_done_prev_rec)

        return runner_state, transition

    runner_state, traj_batch = nnx.scan(_env_step,
                                        in_axes=nnx.Carry,
                                        out_axes=(nnx.Carry, 0),
                                        length=num_steps)(runner_state)

    _, transition = _env_step(runner_state)

    last_val_batteries = transition.values_batteries
    last_val_rec = transition.value_rec

    return runner_state, traj_batch, last_val_batteries, last_val_rec

def compute_battery_actions(runner_state, config, network_batteries, deterministic_batteries):
    rng, _rng = jax.random.split(runner_state.rng)

    actions_batteries = []

    if config['NUM_RL_AGENTS'] > 0:
        last_obs_batteries_rl_num_batteries_first = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1)[:config['NUM_RL_AGENTS']], runner_state.last_obs_batteries)

        if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
            prev_act_state, prev_cri_state = jax.tree.map(lambda x, y: jnp.where(
                runner_state.done_prev_batteries[(slice(None), slice(None)) + (None,) * (x.ndim - 1)], x[None, :], y),
                                                          network_batteries.get_initial_lstm_state(),
                                                          (runner_state.last_lstm_state_batteries.act_state,
                                                           runner_state.last_lstm_state_batteries.cri_state))
            prev_act_state_num_batteries_first, prev_cri_state_num_batteries_first = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1), (prev_act_state, prev_cri_state))
            pi, value_batteries, lstm_act_state, lstm_cri_state = network_batteries(
                last_obs_batteries_rl_num_batteries_first, prev_act_state_num_batteries_first,
                prev_cri_state_num_batteries_first)
            lstm_act_state_batteries, lstm_cri_state_batteries = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1),
                                                                              (lstm_act_state, lstm_cri_state))
        else:
            pi, value_batteries = network_batteries(last_obs_batteries_rl_num_batteries_first)
            lstm_act_state_batteries, lstm_cri_state_batteries = runner_state.last_lstm_state_batteries.act_state, runner_state.last_lstm_state_batteries.cri_state

        actions_batteries_rl = pi.mean() if deterministic_batteries else pi.sample(seed=_rng)  # batteries first
        log_prob_batteries = pi.log_prob(actions_batteries_rl)  # batteries first

        value_batteries, actions_batteries_rl, log_prob_batteries = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1),
                                                                                 (value_batteries, actions_batteries_rl,
                                                                                  log_prob_batteries))  # num_envs first

        actions_batteries.append(actions_batteries_rl)
    else:
        value_batteries = jnp.zeros((config['NUM_ENVS'],))
        log_prob_batteries = jnp.zeros((config['NUM_ENVS'],))
        lstm_act_state_batteries, lstm_cri_state_batteries = runner_state.last_lstm_state_batteries.act_state, runner_state.last_lstm_state_batteries.cri_state

    if config['NUM_BATTERY_FIRST_AGENTS'] > 0:
        idx_start_bf = config['NUM_RL_AGENTS']
        idx_end_bf = config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS']

        demand = runner_state.last_obs_batteries['demand'][:, idx_start_bf:idx_end_bf]
        generation = runner_state.last_obs_batteries['generation'][:, idx_start_bf:idx_end_bf]

        actions_batteries_battery_first = (
                                                      generation - demand) / runner_state.env_state.battery_states.electrical_state.v[
                                                                             :, idx_start_bf:idx_end_bf]

        actions_batteries_battery_first = jnp.expand_dims(actions_batteries_battery_first, -1)

        actions_batteries.append(actions_batteries_battery_first)

    if config['NUM_ONLY_MARKET_AGENTS'] > 0:
        actions_batteries_only_market = jnp.zeros(
            (config['NUM_ENVS'], config['NUM_ONLY_MARKET_AGENTS'], config['BATTERY_ACTION_SPACE_SIZE']))
        actions_batteries.append(actions_batteries_only_market)

    if config['NUM_RANDOM_AGENTS'] > 0:
        rng, _rng = jax.random.split(rng)

        actions_batteries_random = jax.random.uniform(_rng,
                                                      shape=(config['NUM_ENVS'], config['NUM_RANDOM_AGENTS']),
                                                      minval=-1., maxval=1.)

        actions_batteries_random *= config['MAX_ACTION_RANDOM_AGENTS']

        actions_batteries_random = jnp.expand_dims(actions_batteries_random, -1)

        actions_batteries.append(actions_batteries_random)

    actions_batteries = jnp.concat(actions_batteries, axis=1)

    runner_state = runner_state._replace(rng=rng,
                                         last_lstm_state_batteries=LSTMState(lstm_act_state_batteries,
                                                                             lstm_cri_state_batteries)
                                         )

    return runner_state, actions_batteries, value_batteries, log_prob_batteries

def compute_rec_action(runner_state, config, rec_obsv, network_rec, deterministic_rec):
    rng, _rng = jax.random.split(runner_state.rng)

    # defaults
    lstm_act_state_rec, lstm_cri_state_rec = runner_state.last_lstm_state_rec.act_state, runner_state.last_lstm_state_rec.cri_state
    log_probs_rec = jnp.zeros((config['NUM_ENVS'],))
    value_rec = jnp.zeros((config['NUM_ENVS'],))

    if config.get('USE_REC_RULE_BASED_POLICY', False):
        # Rule-based policy
        actions_rec = rec_rule_based_policy(rec_obsv, config['REC_RULE_BASED_NAME'], _rng)
    else:
        net_type_rec = config['NETWORK_TYPE_REC']

        if net_type_rec == 'mlp':
            actions_rec = network_rec(rec_obsv)
        else:
            if net_type_rec == 'recurrent_actor_critic':
                # Reset LSTM state when episode ended
                init_act_state, init_cri_state = network_rec.get_initial_lstm_state()
                prev_act_state, prev_cri_state = jax.tree.map(
                    lambda init, prev: jnp.where(
                        runner_state.done_prev_rec[(slice(None),) + (None,) * init.ndim],
                        init[None, :],
                        prev
                    ),
                    (init_act_state, init_cri_state),
                    (lstm_act_state_rec, lstm_cri_state_rec)
                )

                pi, value_rec, lstm_act_state_rec, lstm_cri_state_rec = network_rec(rec_obsv, prev_act_state,
                                                                                    prev_cri_state)

            else:
                # Non-recurrent actor-critic with value
                pi, value_rec = network_rec(rec_obsv)

            actions_rec = pi.mean() if deterministic_rec else pi.sample(seed=_rng)
            log_probs_rec = pi.log_prob(actions_rec + 1e-8)

    runner_state = runner_state._replace(rng=rng,
                                         last_lstm_state_rec=LSTMState(lstm_act_state_rec,
                                                                       lstm_cri_state_rec))

    return runner_state, actions_rec, value_rec, log_probs_rec