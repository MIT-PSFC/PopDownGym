"""Test the policy trained using ES."""
import equinox as eqx
import jax
import matplotlib.pyplot as plt

import wandb
from scripts.es.train.es_closed_loop import (MLP, create_env,
                                             rollout_closed_loop)


def plot_rollout_results(t, states, reward_inputs, rewards):
    # Plot the state trajectories
    for var, state in states.items():
        fig, ax = plt.subplots()
        ax.plot(t.T, state.T)
        ax.set_ylabel(var)
        ax.set_xlabel("Time (s)")
        wandb.log({f"State Trajectory/{var}": fig}, commit=False)

    for var, term in reward_inputs.items():
        fig, ax = plt.subplots()
        ax.plot(t.T, term.T)

        if var in env.reward_model.limits:
            ax.plot(
                t.T, t.T * 0 + env.reward_model.limits[var], color="k", linestyle="--"
            )

        if var == "Ip_MA":
            ax.plot(
                t.T,
                t.T * 0 + env.reward_model.ip_ma["target"],
                color="k",
                linestyle="--",
            )

        ax.set_ylabel(var)
        ax.set_xlabel("Time (s)")
        wandb.log({f"Reward Input Trajectory/{var}": fig}, commit=False)

    fig, ax = plt.subplots()
    ax.plot(t.T, reward_traces.T)
    ax.set_ylabel("Reward")
    ax.set_xlabel("Time (s)")
    wandb.log({"Reward Trace": fig}, commit=False)

if __name__ == "__main__":
    prng_key = jax.random.PRNGKey(0)

    # Load the environment
    env, _ = create_env()

    # Load the best policy trained using ES
    hidden_dims = 512
    hidden_layers = 4
    prng_key, policy_key = jax.random.split(prng_key)
    mlp = MLP(policy_key, env.n_obs, hidden_layers, hidden_dims, env.n_actions)
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
