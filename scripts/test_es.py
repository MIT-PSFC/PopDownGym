"""Test the policy trained using ES."""

import os

import equinox as eqx
import evosax as es
import jax
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from train_es import MLP, create_env, rollout_closed_loop

from pop_down_gym.model import Model
from pop_down_gym.pd_gym_stateless import PopDownGymStateless

if __name__ == "__main__":
    prng_key = jax.random.PRNGKey(0)

    # Load the environment
    env = create_env()

    # Load the best policy trained using ES
    hidden_dims = 256
    prng_key, policy_key = jax.random.split(prng_key)
    mlp = MLP(policy_key, env.n_obs, hidden_dims, env.n_actions)
    mlp = eqx.tree_deserialise_leaves("tmp/es_best_policy.eqx", mlp)

    # Simulate the policy over a bunch of random rollouts
    n_trials = 10
    steps = 100
    keys = jax.random.split(prng_key, n_trials)
    rollout_fn = lambda key, policy: rollout_closed_loop(key, env, policy, steps=steps)
    rewards, states, t, reward_inputs, reward_traces = jax.vmap(
        rollout_fn, in_axes=(0, None)
    )(keys, mlp)

    # Plot the state trajectories
    n_plots = max(len(list(states.keys())), len(list(reward_inputs.keys())))
    fig, axs = plt.subplots(3, n_plots, figsize=(n_plots * 15, 10))
    for i, (var, state) in enumerate(states.items()):
        axs[0, i].plot(t.T, state.T)
        axs[0, i].set_title(var)

    for i, (var, term) in enumerate(reward_inputs.items()):
        axs[1, i].plot(t.T, term.T)
        axs[1, i].set_title(var)

        if var in env.reward_model.limits:
            axs[1, i].axhline(env.reward_model.limits[var], color="k", linestyle="--")

        if var == "Ip_MA":
            axs[1, i].axhline(
                env.reward_model.ip_ma["target"], color="k", linestyle="--"
            )
        
    axs[2, 0].plot(t.T, reward_traces.T)
    axs[2, 0].set_title("Reward")

    fig.savefig("tmp/es_best_policy_rollout.png")
    plt.close(fig)

    # Plot the reward distribution
    fig, ax = plt.subplots()
    ax.hist(rewards, bins=100)
    ax.set_title("Reward distribution")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Frequency")
    fig.savefig("tmp/es_best_policy_reward_dist.png")
    plt.close(fig)
