import jax
import jax.numpy as jnp


def calculate_gae_batteries(traj_batch, last_val_batteries, config):

    def _get_advantages_batteries(gae_and_next_value, transition_data):
        gae, next_value = gae_and_next_value
        done, value, rewards = transition_data

        # delta = rewards + config['GAMMA'] * next_value * (1 - done) - value
        # gae = (delta + config['GAMMA'] * config['GAE_LAMBDA'] * (1 - done) * gae)

        delta = rewards + config['GAMMA_BATTERIES'] * next_value - value
        gae = (delta + config['GAMMA_BATTERIES'] * config['GAE_LAMBDA'] * gae)

        return (gae, value), gae

    if config['NUM_RL_AGENTS'] > 0:
        rewards_batteries = traj_batch.rewards_batteries[..., :config['NUM_RL_AGENTS']]

        assert rewards_batteries.shape[1] == config['NUM_ENVS']
        assert rewards_batteries.shape[2] == config['NUM_RL_AGENTS']

        if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_BATTERIES']:
            rewards_batteries = (rewards_batteries - rewards_batteries.mean(axis=(0, 1), keepdims=True)) / (
                        rewards_batteries.std(axis=(0, 1), keepdims=True) + 1e-8)

        _, advantages_batteries = jax.lax.scan(
            _get_advantages_batteries,
            (jnp.zeros_like(last_val_batteries), last_val_batteries),
            (traj_batch.done_batteries[..., :config['NUM_RL_AGENTS']], traj_batch.values_batteries[..., :config['NUM_RL_AGENTS']], rewards_batteries),
            reverse=True,
            unroll=32,
        )
        targets_batteries = advantages_batteries + traj_batch.values_batteries

    else:
        advantages_batteries = 0.
        targets_batteries = 0.

    return advantages_batteries, targets_batteries

def calculate_gae_rec(traj_batch, last_val_rec, config):

    def _get_advantages_rec(gae_and_next_value, transition_data):
        gae, next_value = gae_and_next_value
        done, value, rewards = transition_data

        # delta = rewards + config['GAMMA'] * next_value * (1 - done) - value
        # gae = (delta + config['GAMMA'] * config['GAE_LAMBDA'] * (1 - done) * gae)

        delta = rewards + config['GAMMA_REC'] * next_value - value
        gae = (delta + config['GAMMA_REC'] * config['GAE_LAMBDA'] * gae)

        return (gae, value), gae

    if not config.get('USE_REC_RULE_BASED_POLICY', False):
        reward_rec = traj_batch.reward_rec

        if config['NORMALIZE_REWARD_FOR_GAE_AND_TARGETS_REC']:
            reward_rec = (reward_rec - reward_rec.mean()) / (reward_rec.std() + 1e-8)

        _, advantages_rec = jax.lax.scan(
            _get_advantages_rec,
            (jnp.zeros_like(last_val_rec), last_val_rec),
            (traj_batch.done_rec, traj_batch.value_rec, reward_rec),
            reverse=True,
            unroll=32,
        )
        targets_rec = advantages_rec + traj_batch.value_rec

    else:
        advantages_rec = 0.
        targets_rec = 0.

    return advantages_rec, targets_rec
