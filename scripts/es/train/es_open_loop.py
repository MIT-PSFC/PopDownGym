"""
Train an open-loop (feedforward controls) using evolutionary strategy (ES).

Sweep a range of uncertainties (width of parameter bounds). For each:
    - Train a trajectory using ES
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
from scipy.special import binom
from tqdm import tqdm

import wandb
from pop_down_gym.pd_gym_stateless import PopDownGymStateless
from scripts.es.train.es_closed_loop import (plot_action_trajectory,
                                             plot_hit_time_vs_reward,
                                             plot_test_set_trajectories)


class MLP(eqx.Module):
    layers: list

    def __init__(self, key, hidden_layers, hidden_dims, output_dims):
        keys = jax.random.split(key, hidden_layers)
        self.layers = [eqx.nn.Linear(1, hidden_dims, key=keys[0])]
        for key in keys[1:-1]:
            self.layers.append(eqx.nn.Linear(hidden_dims, hidden_dims, key=key))
        self.layers.append(eqx.nn.Linear(hidden_dims, output_dims, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jax.nn.tanh(self.layers[-1](x))


class Spline(eqx.Module):
    """
    A fixed-degree Bezier curve function of time.

    Time is normalized to [0, 1]

    args:
        p: the array of control points for the Bezier curve
    """

    def __init__(self, key, degree, dimension):
        # Generate some random initial control points
        self.p = jax.random.normal(key, (degree, dimension))

    @property
    def n(self):
        """Return the degree of this Bezier curve"""
        return self.p.shape[0] - 1

    @property
    def i(self):
        """Return the degree indices"""
        return np.arange(self.n + 1)

    @property
    def coefficients(self):
        """Return the degree indices"""
        return binom(self.n, self.i)

    @jax.jit
    def __call__(self, t):
        """
        Return the point along the trajectory at the given time.
        
        Args:
            t: the normalized time to evaluate the spline at (between 0 and 1)
        """
        # Bezier curves have an explicit form
        # see https://en.wikipedia.org/wiki/B%C3%A9zier_curve
        return jnp.sum(
            self.coefficients * (1 - t) ** (self.n - self.i) * t**self.i * self.p.T,
            axis=-1,
        )

def rollout_open_loop(prng_key, env, trajectory_fn, steps=100):
    """Simulate the trajectory in the environment."""
    # Sample random parameters and initial state
    params, initial_state, initial_obs, _ = env.reset(prng_key)

    # Define a step function to simulate using scan
    def scan_step(carry, _):
        # Unpack the carry
        state, obs, t, done = carry

        # Vectorize observation
        obs = jax.numpy.hstack((obs["continuous"], obs["Hmode"]))

        # Get the action
        action = trajectory_fn(t.reshape(1) / (steps * env.dt))

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
    _, (rewards, states, t, reward_inputs, hit_goal, actions) = jax.lax.scan(
        scan_step, (initial_state, initial_obs, 0.0, False), None, length=steps
    )

    return rewards.sum(), states, t, reward_inputs, rewards, hit_goal, actions


def train_es_open_loop(
    uncertainty_size: float,
    hidden_dims: int = 512,
    hidden_layers: int = 2,
    spine_degree: int = 5,
    use_spline=True,
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
    rollout_test = lambda key, traj: rollout_open_loop(
        key, test_env, traj, simulation_steps
    )

    # Init wandb and save hyperparams
    wandb.init(
        project="popdown",
        name="es-open-loop",
        config={
            "hidden_dims": hidden_dims,
            "hidden_layers": hidden_layers,
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
        "open_loop_spline",
        f"uncertainty_{uncertainty_size:.2f}",
        f"lr_{lrate_init:.1e}",
    )
    os.makedirs(save_path, exist_ok=True)

    # Define the fitness function
    fitness_single_rollout = lambda key, traj: rollout_open_loop(key, env, traj)[0]

    def fitness_multiple_rollouts(key, traj):
        keys = jax.random.split(key, num_eval_rollouts)
        return jax.vmap(fitness_single_rollout, in_axes=(0, None))(keys, traj)

    mean_fitness_multiple_rollouts = lambda key, traj: fitness_multiple_rollouts(
        key, traj
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

    prng_key, policy_key = jax.random.split(prng_key)
    if use_spline:
        initial_traj = Spline(policy_key, spine_degree, env.n_actions)
    else:
        # Create an MLP to represent the trajectory
        initial_traj = MLP(policy_key, hidden_layers, hidden_dims, env.n_actions)

    # Set up ES
    param_reshaper = es.ParameterReshaper(initial_traj)
    one_device_reshaper = es.ParameterReshaper(initial_traj, n_devices=1)
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
            best_trajectory = jax.tree_map(
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
            ) = jax.vmap(rollout_test, in_axes=(0, None))(keys, best_trajectory)

            plot_test_set_trajectories(
                t_test, reward_inputs_test, save_path, commit_wandb=False
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

    # Get the best trajectory
    best_trajectory = jax.tree_map(
        lambda x: x[0], one_device_reshaper.reshape(log["top_params"])
    )

    # Get the state trajectories, reward inputs, and reward distribution on the training
    # uncertainty range
    keys = jax.random.split(prng_key, num_eval_rollouts)
    rollout_train = lambda key, traj: rollout_open_loop(
        key, env, traj, simulation_steps
    )
    (
        rewards_train,
        states_train,
        t_train,
        reward_inputs_train,
        _,
        _,
        _,
    ) = jax.vmap(
        rollout_train, in_axes=(0, None)
    )(keys, best_trajectory)

    # Get the state trajectories, reward inputs, and reward distribution on the full
    # uncertainty range
    test_env = PopDownGymStateless.create_env()
    rollout_test = lambda key, traj: rollout_open_loop(
        key, test_env, traj, simulation_steps
    )
    (
        rewards_test,
        states_test,
        t_test,
        reward_inputs_test,
        _,
        hit_goal_test,
        actions_test,
    ) = jax.vmap(rollout_test, in_axes=(0, None))(keys, best_trajectory)

    # Save experiment parameters
    eqx.tree_serialise_leaves(
        os.path.join(save_path, "config.eqx"),
        {
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
    eqx.tree_serialise_leaves(
        os.path.join(save_path, "best_trajectory.eqx"), best_trajectory
    )
    wandb.save(os.path.join(save_path, "best_trajectory.eqx"))

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
    plot_test_set_trajectories(t_test, reward_inputs_test, save_path, commit_wandb=True)

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
        train_es_open_loop(uncertainty_size)
