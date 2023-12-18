"""
Train a closed-loop (MLP) policy using evolutionary strategy (ES).

Sweep a range of uncertainties (width of parameter bounds). For each:
    - Train a policy using ES
    - Simulate the policy over a test set with the same uncertainty
    - Simulate the policy over a test set with full uncertainty
    - Save the policy, trajectories, reward inputs, the reward distributions over both
        test sets.
"""
import os

import equinox as eqx
import evosax as es
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import wandb
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


class MLP(eqx.Module):
    layers: list

    def __init__(self, key, input_dims, hidden_layers, hidden_dims, output_dims):
        keys = jax.random.split(key, hidden_layers)
        self.layers = [eqx.nn.Linear(input_dims, hidden_dims, key=keys[0])]
        for key in keys[1:-1]:
            self.layers.append(eqx.nn.Linear(hidden_dims, hidden_dims, key=key))
        self.layers.append(eqx.nn.Linear(hidden_dims, output_dims, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return jax.nn.tanh(self.layers[-1](x))


def rollout_closed_loop(prng_key, env, policy, steps=100):
    """Simulate the policy in the environment."""
    # Sample random parameters and initial state
    params, initial_state, initial_obs, _ = env.reset(prng_key)

    # Define a step function to simulate using scan
    def scan_step(carry, _):
        # Unpack the carry
        state, obs, t, done = carry

        # Vectorize observation
        obs = jax.numpy.hstack((obs["continuous"], obs["Hmode"]))

        # Evaluate the policy
        action = policy(obs)

        # Step the environment
        obs, reward, terminated, _, info = env.step(t, params, state, action)

        # If we've terminated, don't update the state
        next_state = jax.lax.cond(
            terminated, lambda _: state, lambda _: info["state"], None
        )
        next_time = jax.lax.cond(terminated, lambda _: t, lambda _: info["time"], None)

        # If we've already terminated (last step), don't update the reward
        reward = jax.lax.cond(done, lambda _: 0.0, lambda _: reward, None)
        done = jnp.logical_or(done, terminated)

        # prepare the carry for the next iteration
        carry = (next_state, obs, next_time, done)
        output = (
            reward,
            next_state,
            next_time,
            info["reward_inputs"],
            info["hit_goal"],
            action,
        )

        return carry, output

    # Simulate the trajectory
    _, (rewards, states, t, reward_inputs, hit_goal, action) = jax.lax.scan(
        scan_step, (initial_state, initial_obs, 0.0, False), None, length=steps
    )

    return rewards.sum(), states, t, reward_inputs, rewards, hit_goal, action


def plot_action_trajectory(
    env, t, action_trajectory, save_path=None, commit_wandb=False
):
    action_lims = {
        "dIp_dt": [-3.0, -0.5],
        "dPaux_dt": [-5.0, 5.0],
        "fueling19": [0.0, 10.0],
        "dgs_dt": [0.0, 1.0],
    }

    unnormalized_trajectory = jax.vmap(jax.vmap(env.dictify_and_unnormalize_action))(
        action_trajectory
    )
    n_actions = env.n_actions
    figsize = np.array([6, 1.2 * n_actions])
    fig, axes = plt.subplots(
        n_actions, layout="constrained", figsize=figsize, sharex=True, dpi=350
    )
    for i, label in enumerate(unnormalized_trajectory):
        ax = axes[i]
        actions = unnormalized_trajectory[label]
        ax.plot(t.T, actions.T, color="C1", lw=0.5, alpha=0.4)
        ax.set_ylabel(label, rotation=0, ha="right")

        # Set the limits.
        ax.autoscale_view()
        ax.set_ylim(action_lims[label])

        # Plot the limits.
        ymin, ymax = ax.get_ylim()
        # Expand the ymax a bit.
        yrange = ymax - ymin
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ymin, ymax = ax.get_ylim()
        # Add a shaded region indicating the limit.
        ax.axhspan(min(ymax, action_lims[label][1]), ymax, color="C0", alpha=0.2)
        ax.axhspan(ymin, max(ymin, action_lims[label][0]), color="C0", alpha=0.2)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "action_trajectory.png"))

    wandb.log({"Feedforward trajectory": fig}, commit=commit_wandb)


def plot_test_set_trajectories(env, t, reward_inputs, save_path=None, commit_wandb=False):
    constr_labels = PopDownGymStateless.constr_labels()
    nconstr = len(constr_labels)
    ylims = {
        "Ip_MA": [1.2e00, 9.18e00],
        "Bv_dot_mag": [5.14e-02, 3.30e-01],
        "Wdot_mag": [-1.42e06, 4.93e07],
        "beta_n": [1.25e-03, 1.17e-02],
        "beta_p": [6.96e-02, 4.57e-01],
        "li": [3.36e-01, 4.19e00],
        "ng_frac": [2.30e-01, 7.83e-01],
        "shafranov_coeff": [1.95, 4.05],
        "iota95": [0.05, 0.25],
    }
    constr_ub = env.reward_model.limits
    Ip_MA_tgt = 2.0

    figsize = np.array([6, 1.2 * nconstr])
    fig, axes = plt.subplots(
        nconstr, layout="constrained", figsize=figsize, sharex=True, dpi=350
    )
    for i, ax in enumerate(axes):
        label = constr_labels[i]
        bT_rew_input = reward_inputs[label]
        ax.plot(t.T, bT_rew_input.T, color="C1", lw=0.5, alpha=0.4)
        ax.set_ylabel(label, rotation=0, ha="right")

        # Set the limits.
        ax.autoscale_view()
        ax.set_ylim(ylims[label])

        # Plot the limits.
        ymin, ymax = ax.get_ylim()
        if label in constr_ub:
            # Make sure the limit is within the plot limits.
            ymin = min(ymin, constr_ub[label])
            ymax = max(ymax, constr_ub[label])

            # Expand the ymax a bit.
            yrange = ymax - ymin
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            ymin, ymax = ax.get_ylim()

            ax.axhspan(min(ymax, constr_ub[label]), ymax, color="C0", alpha=0.2)

        if label == "Ip_MA":
            ax.axhspan(ymin, Ip_MA_tgt, color="C5", alpha=0.2)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "test_env_performance.png"))

    wandb.log({"Test env trajectory": fig}, commit=commit_wandb)


def plot_hit_time_vs_reward(t, hit_goal, rewards, save_path=None, commit_wandb=False):
    episode_lengths = jnp.max(t, axis=-1)
    hit_goal_at_episode_end = hit_goal[:, -1]
    fig = plt.figure(figsize=(6, 6), dpi=350)
    ax = fig.add_subplot(111)
    ax.scatter(
        episode_lengths[hit_goal_at_episode_end],
        rewards[hit_goal_at_episode_end],
        marker=".",
        color="C0",
        alpha=0.5,
        label="Reached goal",
    )
    ax.scatter(
        episode_lengths[~hit_goal_at_episode_end],
        rewards[~hit_goal_at_episode_end],
        marker=".",
        color="C1",
        alpha=0.5,
        label="Did not reach goal",
    )
    ax.set_xlabel("Episode length (s)")
    ax.set_ylabel("Reward")
    ax.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "episode_length_vs_reward.png"))

    wandb.log({"Episode length vs reward": fig}, commit=commit_wandb)


def train_es_closed_loop(
    uncertainty_size: float,
    hidden_dims: int = 512,
    hidden_layers: int = 2,
    simulation_steps: int = 100,
    num_generations: int = 100,
    top_k: int = 5,
    popsize: int = int(4e1),
    num_eval_rollouts: int = int(1e3),
    lrate_init: float = 1e-2,
    plot_every: int = 10,
):
    # Set the seed for reproducibility
    prng_key = jax.random.PRNGKey(0)

    # Load the environment
    env = PopDownGymStateless.create_env()

    # Overwrite the uncertainty set with the adjusted width
    uncertainty_size = jnp.clip(uncertainty_size, 0.0, 1.0)
    param_ranges = {
        key: ((lb + ub) / 2.0, (ub - lb) * uncertainty_size)
        for key, (lb, ub) in env.random_param_ranges.items()
    }
    env.random_param_ranges = {
        key: (center - width / 2.0, center + width / 2.0)
        for key, (center, width) in param_ranges.items()
    }

    # Also create a test env with full uncertainty
    test_env = PopDownGymStateless.create_env()
    rollout_test = lambda key, traj: rollout_closed_loop(key, test_env, traj)

    # Init wandb and save hyperparams
    wandb.init(
        project="popdown",
        name="es-closed-loop",
        config={
            "hidden_layers": hidden_layers,
            "hidden_dims": hidden_dims,
            "simulation_steps": simulation_steps,
            "num_generations": num_generations,
            "top_k": top_k,
            "popsize": popsize,
            "num_eval_rollouts": num_eval_rollouts,
            "lrate_init": lrate_init,
            "reward_model": env.reward_model.params,
            "uncertainty_size": uncertainty_size,
            "uncertainty_set": env.random_param_ranges,
        },
    )
    save_path = os.path.join(
        "tmp",
        "es",
        "closed_loop",
        f"uncertainty_{uncertainty_size:.2f}",
        f"lr_{lrate_init:.1e}",
    )
    os.makedirs(save_path, exist_ok=True)

    # Define the fitness function
    fitness_single_rollout = lambda key, policy: rollout_closed_loop(
        key, env, policy, steps=simulation_steps
    )[0]

    def fitness_multiple_rollouts(key, policy):
        keys = jax.random.split(key, num_eval_rollouts)
        return jax.vmap(fitness_single_rollout, in_axes=(0, None))(keys, policy)

    mean_fitness_multiple_rollouts = lambda key, policy: fitness_multiple_rollouts(
        key, policy
    ).mean()

    def population_fitness(key, population):
        # If there is only one device (no leading axis), then don't pmap
        if jax.local_device_count() == 1:
            return jax.vmap(mean_fitness_multiple_rollouts, in_axes=(None, 0))(
                key, population
            )
        else:
            # The population has a leading axis for devices and a second axis
            # for vectorized dimension
            return jax.pmap(
                jax.jit(jax.vmap(mean_fitness_multiple_rollouts, in_axes=(None, 0))),
                in_axes=(None, 0),
            )(key, population)

    # Create an MLP
    prng_key, policy_key = jax.random.split(prng_key)
    mlp = MLP(policy_key, env.n_obs, hidden_layers, hidden_dims, env.n_actions)

    # Set up ES
    param_reshaper = es.ParameterReshaper(mlp)
    one_device_reshaper = es.ParameterReshaper(mlp, n_devices=1)
    strategy = es.OpenES(
        popsize=popsize,
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        lrate_init=lrate_init,
    )
    es_logging = es.ESLog(
        param_reshaper.total_params,
        num_generations=num_generations,
        top_k=top_k,
        maximize=True,
    )
    log = es_logging.initialize()
    fit_shaper = es.FitnessShaper(z_score=True, w_decay=0.0, maximize=True)
    prng_key, es_key = jax.random.split(prng_key)
    state = strategy.initialize(es_key)

    # Run the evolution
    pbar = tqdm(range(num_generations))
    for gen in pbar:
        prng_key, prng_ask, prng_eval = jax.random.split(prng_key, 3)

        # Generate a new population
        x, state = strategy.ask(prng_ask, state)

        # Evaluate the population
        reshaped_params = param_reshaper.reshape(x)
        fitness = population_fitness(prng_eval, reshaped_params).reshape(-1)
        fit_re = fit_shaper.apply(x, fitness)

        # Update the population
        state = strategy.tell(x, fit_re, state)

        # Log
        log = es_logging.update(log, x, fitness)
        pbar.set_description(f"Current gen top fitness: {log['log_gen_1'][gen]:.3f}")

        # Plot the top trajectory
        if gen % plot_every == 0:
            best_policy = jax.tree_map(
                lambda x: x[0], one_device_reshaper.reshape(log["top_params"])
            )
            prng_key, prng_eval = jax.random.split(prng_key)
            keys = jax.random.split(prng_eval, 50)
            (
                rewards_test,
                states_test,
                t_test,
                reward_inputs_test,
                _,
                hit_goal_test,
                actions_test,
            ) = jax.vmap(rollout_test, in_axes=(0, None))(keys, best_policy)

            plot_test_set_trajectories(
                env, t_test, reward_inputs_test, save_path, commit_wandb=False
            )
            plot_action_trajectory(
                env, t_test, actions_test, save_path, commit_wandb=False
            )
            plot_hit_time_vs_reward(
                t_test, hit_goal_test, rewards_test, save_path, commit_wandb=False
            )

        wandb.log(
            {
                "Top 1 Fitness": log["log_top_1"][gen],
                f"Top {top_k} Mean Fitness": log["log_top_mean"][gen],
                "Top 1 Fitness (current gen)": log["log_gen_1"][gen],
                "Mean Fitness (current gen)": log["log_gen_mean"][gen],
            }
        )

    # Get the best policy
    top_policies = one_device_reshaper.reshape(log["top_params"])
    best_policy = jax.tree_map(lambda x: x[0], top_policies)

    # Get the state trajectories, reward inputs, and reward distribution on the training
    # uncertainty range
    keys = jax.random.split(prng_key, num_eval_rollouts)
    rollout_train = lambda key, policy: rollout_closed_loop(
        key, env, policy, steps=simulation_steps
    )
    rewards_train, states_train, t_train, reward_inputs_train, _, _, _ = jax.vmap(
        rollout_train, in_axes=(0, None)
    )(keys, best_policy)

    # Get the state trajectories, reward inputs, and reward distribution on the full
    # uncertainty range
    test_env = PopDownGymStateless.create_env()
    rollout_test = lambda key, policy: rollout_closed_loop(
        key, test_env, policy, steps=simulation_steps
    )
    (
        rewards_test,
        states_test,
        t_test,
        reward_inputs_test,
        _,
        hit_goal_test,
        actions_test,
    ) = jax.vmap(rollout_test, in_axes=(0, None))(keys, best_policy)

    # Save experiment parameters
    eqx.tree_serialise_leaves(
        os.path.join(save_path, "config.eqx"),
        {
            "hidden_layers": hidden_layers,
            "hidden_dims": hidden_dims,
            "simulation_steps": simulation_steps,
            "num_generations": num_generations,
            "top_k": top_k,
            "popsize": popsize,
            "num_eval_rollouts": num_eval_rollouts,
            "lrate_init": lrate_init,
            "reward_model": env.reward_model.params,
            "uncertainty_size": uncertainty_size,
            "uncertainty_set": env.random_param_ranges,
        },
    )

    # Save the best policy
    eqx.tree_serialise_leaves(os.path.join(save_path, "best_policy.eqx"), best_policy)
    wandb.save(os.path.join(save_path, "best_policy.eqx"))

    # Save the training set performance
    eqx.tree_serialise_leaves(
        os.path.join(save_path, "training_env_performance.eqx"),
        {
            "rewards": rewards_train,
            "states": states_train,
            "t": t_train,
            "reward_inputs": reward_inputs_train,
        },
    )
    wandb.save(os.path.join(save_path, "training_env_performance.eqx"))

    # Save the test set performance
    eqx.tree_serialise_leaves(
        os.path.join(save_path, "test_env_performance.eqx"),
        {
            "rewards": rewards_test,
            "states": states_test,
            "t": t_test,
            "reward_inputs": reward_inputs_test,
        },
    )
    wandb.save(os.path.join(save_path, "test_env_performance.eqx"))

    # Plot the trajectories on the test set
    plot_test_set_trajectories(env, t_test, reward_inputs_test, save_path, commit_wandb=True)

    # Plot trajectory in unnormalized action space
    plot_action_trajectory(env, t_test, actions_test, save_path, commit_wandb=True)

    # Plot the distribution of hitting time vs reward
    plot_hit_time_vs_reward(
        t_test, hit_goal_test, rewards_test, save_path, commit_wandb=True
    )

    # End the wandb run
    wandb.finish()


if __name__ == "__main__":
    # Run a sweep over the uncertainty size
    uncertainty_sizes = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    # TODO only run one for debugging
    uncertainty_sizes = [1.0]

    for uncertainty_size in uncertainty_sizes:
        train_es_closed_loop(uncertainty_size)
