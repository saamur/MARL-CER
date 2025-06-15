import jax
import jax.numpy as jnp

from flax import nnx
import optax

from algorithms.train_core import RunnerState, UpdateState, Transition, TrainState
from ernestogym.envs.multi_agent.env import RECEnv

from algorithms.rec_rule_based_policies import rec_rule_based_policy


def test_networks(env:RECEnv, train_state:TrainState, num_iter, config, rng, curr_iter=0, print_data=False):

    networks_batteries, network_rec = nnx.merge(train_state.graph_def, train_state.state)

    if config['NUM_RL_AGENTS'] > 0:
        networks_batteries.eval()
    if not config.get('USE_REC_RULE_BASED_POLICY', False):
        network_rec.eval()

    rng, _rng = jax.random.split(rng)

    obsv, env_state = env.reset(_rng, profile_index=0)

    if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic' and config['NUM_RL_AGENTS'] > 0:
        init_act_state_batteries, init_cri_state_batteries = networks_batteries.get_initial_lstm_state()
        act_state_batteries, cri_state_batteries = init_act_state_batteries, init_cri_state_batteries
    else:
        act_state_batteries, cri_state_batteries = None, None

    if not config.get('USE_REC_RULE_BASED_POLICY', False) and config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
        init_act_state_rec, init_cri_state_rec = network_rec.get_initial_lstm_state()
        act_state_rec, cri_state_rec = init_act_state_rec, init_cri_state_rec
    else:
        act_state_rec, cri_state_rec = None, None

    # @scan_tqdm(num_iter, print_rate=num_iter // 100)
    def _env_step(runner_state, unused):
        obsv_batteries, env_state, act_state_batteries, cri_state_batteries, act_state_rec, cri_state_rec, rng, next_profile_index = runner_state

        actions_batteries = []

        if config['NUM_RL_AGENTS'] > 0:

            obsv_batteries_rl = jax.tree.map(lambda x: x[:config['NUM_RL_AGENTS']], obsv_batteries)

            if config['NETWORK_TYPE_BATTERIES'] == 'recurrent_actor_critic':
                pi, value_batteries, act_state_batteries, cri_state_batteries = networks_batteries(obsv_batteries_rl, act_state_batteries, cri_state_batteries)
            else:
                pi, value_batteries = networks_batteries(obsv_batteries_rl)

            #deterministic action
            actions_batteries_rl = pi.mean()

            actions_batteries_rl = actions_batteries_rl.squeeze(axis=-1)
            actions_batteries.append(actions_batteries_rl)


        if config['NUM_BATTERY_FIRST_AGENTS'] > 0:
            idx_start_bf = config['NUM_RL_AGENTS']
            idx_end_bf = config['NUM_RL_AGENTS'] + config['NUM_BATTERY_FIRST_AGENTS']

            demand = obsv_batteries['demand'][idx_start_bf:idx_end_bf]
            generation = obsv_batteries['generation'][idx_start_bf:idx_end_bf]

            actions_batteries_battery_first = (generation - demand) / env_state.battery_states.electrical_state.v[idx_start_bf:idx_end_bf]

            actions_batteries.append(actions_batteries_battery_first)

        if config['NUM_ONLY_MARKET_AGENTS'] > 0:
            actions_batteries_only_market = jnp.zeros(
                (config['NUM_ONLY_MARKET_AGENTS'],))
            actions_batteries.append(actions_batteries_only_market)

        if config['NUM_RANDOM_AGENTS'] > 0:
            rng, _rng = jax.random.split(rng)

            actions_batteries_random = jax.random.uniform(_rng,
                                                          shape=(config['NUM_RANDOM_AGENTS'],),
                                                          minval=-1.,
                                                          maxval=1.)

            actions_batteries_random *= config['MAX_ACTION_RANDOM_AGENTS']

            actions_batteries.append(actions_batteries_random)

        actions_batteries = jnp.concat(actions_batteries, axis=0)

        actions_first = {env.battery_agents[i]: actions_batteries[i] for i in range(env.num_battery_agents)}
        actions_first[env.rec_agent] = jnp.zeros(env.num_battery_agents)

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward_first, done_first, info_first = env.step(_rng, env_state, actions_first)

        rec_obsv = obsv[env.rec_agent]

        if config.get('USE_REC_RULE_BASED_POLICY', False):
            actions_rec = rec_rule_based_policy(rec_obsv, config['REC_RULE_BASED_NAME'], _rng)
        else:
            net_type_rec = config['NETWORK_TYPE_REC']
            if net_type_rec == 'mlp':
                actions_rec = network_rec(rec_obsv)
            else:
                if config['NETWORK_TYPE_REC'] == 'recurrent_actor_critic':
                    pi, _, act_state_rec, cri_state_rec = network_rec(rec_obsv, act_state_rec, cri_state_rec)
                else:
                    pi, _ = network_rec(rec_obsv)
                actions_rec = pi.mean()

        actions_second = {agent: jnp.array(0.) for agent in env.battery_agents}
        actions_second[env.rec_agent] = actions_rec

        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward_second, done_second, info_second = env.step(_rng, env_state, actions_second)

        done = jnp.logical_or(done_first['__all__'], done_second['__all__'])

        info = jax.tree.map(lambda  x, y: x + y, info_first, info_second)

        info['actions_batteries'] = actions_batteries
        info['actions_rec'] = actions_rec
        info['dones'] = jax.tree.map(lambda x, y : jnp.logical_or(x, y), done_first, done_second)

        rng, _rng = jax.random.split(rng)
        obsv, env_state,next_profile_index = jax.lax.cond(done,
                                                          lambda : env.reset(_rng, profile_index=next_profile_index) + (next_profile_index+1,),
                                                          lambda : (obsv, env_state, next_profile_index))

        obs_batteries = jax.tree.map(lambda *vals: jnp.stack(vals), *[obsv[a] for a in env.battery_agents])

        runner_state = (obs_batteries, env_state, act_state_batteries, cri_state_batteries, act_state_rec, cri_state_rec, rng, next_profile_index)
        return runner_state, info

    obsv_batteries = jax.tree.map(lambda *vals: jnp.stack(vals), *[obsv[a] for a in env.battery_agents])

    runner_state = (obsv_batteries, env_state, act_state_batteries, cri_state_batteries, act_state_rec, cri_state_rec, rng, 1)

    runner_state, info = jax.lax.scan(_env_step, runner_state, jnp.arange(num_iter))

    reward_type = 'weig_reward'

    if print_data:

        jax.debug.print('curr_iter: {i}', i=curr_iter)
        for i in range(config['NUM_BATTERY_AGENTS']):
            jax.debug.print(
                '\tr_tot: {r_tot}\n\tr_trad: {r_trad}\n\tr_deg: {r_deg}\n\tr_clip: {r_clip}\n\tr_glob: {r_glob}\n'
                '\tmean soc: {mean_soc}\n',
                r_tot=jnp.sum(info['r_tot'][:, i]),
                r_trad=jnp.sum(info[reward_type]['r_trad'][:, i]),
                r_deg=jnp.sum(info[reward_type]['r_deg'][:, i]),
                r_clip=jnp.sum(info[reward_type]['r_clipping'][:, i]),
                r_glob=jnp.sum(info[reward_type]['r_glob'][:, i]),
                mean_soc=jnp.mean(info['soc'][:, i]))

        jax.debug.print('\n\tstd actions: {std_act}\n\tself consumption: {sc}\n\treward REC: {rec_rew}\n',
                        std_act=jnp.std(info['actions_batteries'], axis=0),
                        sc=jnp.sum(info['self_consumption']),
                        rec_rew=jnp.sum(info['rec_reward']))

        jax.debug.print('\n\tr_tot: {x}', x=jnp.sum(info['r_tot'][:, :config['NUM_RL_AGENTS']]))

    return info