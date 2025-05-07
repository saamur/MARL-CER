import jax
import jax.numpy as jnp
import distrax


def rec_rule_based_policy(rec_obs, name, rng):
    if name == 'scarce_resource':
        return rec_scarce_resource_policy(rec_obs, rng)
    elif name == 'both_resources':
        return rec_both_resources_policy(rec_obs, rng)
    elif name == 'random':
        return rec_random_policy(rec_obs, rng)
    elif name == 'uniform':
        return rec_uniform_policy(rec_obs, rng)
    else:
        raise ValueError('Unknown REC rule-based policy name {}'.format(name))


def rec_scarce_resource_policy(rec_obs, rng):
    net_exchange = rec_obs['generations_battery_houses'] - rec_obs['demands_base_battery_houses'] - rec_obs['demands_battery_battery_houses']

    net_exchange_plus = jnp.maximum(net_exchange, 0)
    net_exchange_minus = -jnp.minimum(net_exchange, 0)

    action_plus = net_exchange_plus / (net_exchange_plus.sum(axis=-1, keepdims=True) + 1e-8)
    action_minus = net_exchange_minus / (net_exchange_minus.sum(axis=-1, keepdims=True) + 1e-8)

    actions = (rec_obs['network_REC_plus'] > rec_obs['network_REC_minus'])[..., None] * action_minus + (rec_obs['network_REC_plus'] <= rec_obs['network_REC_minus'])[..., None] * action_plus

    actions = jnp.where(actions.sum(axis=-1, keepdims=True) == 0., jnp.ones_like(actions)/actions.shape[-1], actions)

    return actions

def rec_both_resources_policy(rec_obs, rng):
    net_exchange = rec_obs['generations_battery_houses'] - rec_obs['demands_base_battery_houses'] - rec_obs['demands_battery_battery_houses']

    net_exchange_plus = jnp.maximum(net_exchange, 0)
    net_exchange_minus = -jnp.minimum(net_exchange, 0)

    frac_plus = net_exchange_plus / (net_exchange_plus.sum(axis=-1, keepdims=True) + 1e-8)
    frac_minus = net_exchange_minus / (net_exchange_minus.sum(axis=-1, keepdims=True) + 1e-8)

    frac_plus = jnp.where(frac_plus.sum(axis=-1, keepdims=True) == 0., jnp.ones_like(frac_plus)/frac_plus.shape[-1], frac_plus)
    frac_minus = jnp.where(frac_minus.sum(axis=-1, keepdims=True) == 0., jnp.ones_like(frac_minus)/frac_minus.shape[-1], frac_minus)

    actions = (frac_plus + frac_minus) / 2

    return actions

def rec_uniform_policy(rec_obs, rng):

    # jax.debug.print('uniform', ordered=True)
    shape = rec_obs['generations_battery_houses'].shape
    actions = jnp.ones(shape)/shape[-1]
    return actions

def rec_random_policy(rec_obs, rng):
    # jax.debug.print('random', ordered=True)
    shape = rec_obs['generations_battery_houses'].shape
    pi = distrax.Dirichlet(jnp.ones(shape[-1]))
    actions = pi.sample(seed=rng, sample_shape=shape[:-1])
    return actions