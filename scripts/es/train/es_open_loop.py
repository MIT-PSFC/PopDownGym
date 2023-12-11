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
from tqdm import tqdm

import wandb
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def rollout_open_loop(prng_key, env, control_trajectory):
    """Simulate the trajectory in the environment."""
    # Sample random parameters and initial state
    params, initial_state, initial_obs, _ = env.reset(prng_key)

    # Define a step function to simulate using scan
    def scan_step(carry, action):
        # Unpack the carry
        state, obs, t, done = carry

        # Vectorize observation
        obs = jax.numpy.hstack((obs["continuous"], obs["Hmode"]))

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
        output = (reward, next_state, next_time, info["reward_inputs"])

        return carry, output

    # Simulate the trajectory
    _, (rewards, states, t, reward_inputs) = jax.lax.scan(
        scan_step, (initial_state, initial_obs, 0.0, False), control_trajectory
    )

    return rewards.sum(), states, t, reward_inputs, rewards


def train_es_open_loop(
    uncertainty_size: float,
    simulation_steps: int = 100,
    num_generations: int = 200,
    top_k: int = 5,
    popsize: int = int(4e1),
    num_eval_rollouts: int = int(1e3),
    lrate_init: float = 1e-2,
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

    # Init wandb and save hyperparams
    wandb.init(
        project="popdown",
        name="es-open-loop",
        config={
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

    # Define the fitness function
    fitness_single_rollout = lambda key, trajectory: rollout_open_loop(
        key, env, trajectory
    )[0]

    def fitness_multiple_rollouts(key, trajectory):
        keys = jax.random.split(key, num_eval_rollouts)
        return jax.vmap(fitness_single_rollout, in_axes=(0, None))(keys, trajectory)

    mean_fitness_multiple_rollouts = lambda key, trajectory: fitness_multiple_rollouts(
        key, trajectory
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

    # Create a trajectory
    prng_key, traj_key = jax.random.split(prng_key)
    trajectory = jax.random.normal(traj_key, (simulation_steps, env.n_actions))

    # Set up ES
    param_reshaper = es.ParameterReshaper(trajectory)
    one_device_reshaper = es.ParameterReshaper(trajectory, n_devices=1)
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
    fit_shaper = es.FitnessShaper(centered_rank=True, w_decay=0.0, maximize=True)
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
        pbar.set_description(f"Performance: {log['log_top_1'][gen]:.3f}")
        wandb.log(
            {
                "Top 1 Fitness": log["log_top_1"][gen],
                f"Top {top_k} Mean Fitness": log["log_top_mean"][gen],
                "Top 1 Fitness (current gen)": log["log_gen_1"][gen],
                "Mean Fitness (current gen)": log["log_gen_mean"][gen],
            }
        )

    # Get the best trajectory
    top_trajectories = one_device_reshaper.reshape(log["top_params"])
    best_trajectory = jax.tree_map(lambda x: x[0], top_trajectories)

    # Get the state trajectories, reward inputs, and reward distribution on the training
    # uncertainty range
    keys = jax.random.split(prng_key, num_eval_rollouts)
    rollout_train = lambda key, trajectory: rollout_open_loop(key, env, trajectory)
    rewards_train, states_train, t_train, reward_inputs_train, _ = jax.vmap(
        rollout_train, in_axes=(0, None)
    )(keys, best_trajectory)

    # Get the state trajectories, reward inputs, and reward distribution on the full
    # uncertainty range
    test_env = PopDownGymStateless.create_env()
    rollout_test = lambda key, trajectory: rollout_open_loop(key, test_env, trajectory)
    rewards_test, states_test, t_test, reward_inputs_test, _ = jax.vmap(
        rollout_test, in_axes=(0, None)
    )(keys, best_trajectory)

    # Save experiment parameters
    save_path = os.path.join(
        "tmp", "es", "open_loop", f"uncertainty_{uncertainty_size:.2f}"
    )
    os.makedirs(save_path, exist_ok=True)
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


if __name__ == "__main__":
    # Run a sweep over the uncertainty size
    uncertainty_sizes = jnp.linspace(0.0, 1.0, 11)

    for uncertainty_size in uncertainty_sizes:
        train_es_open_loop(uncertainty_size)
