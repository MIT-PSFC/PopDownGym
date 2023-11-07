"""Tests and benchmarking for PopDownGym"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from pop_down_gym.model import Model
from pop_down_gym.pd_gym import PopDownGym


def create_env():
    """Create an instance of PopDownGym."""
    config_filepath = os.path.join(os.path.dirname(__file__), "../configs/gym.yaml")
    config = yaml.safe_load(open(config_filepath, "r"))
    model, _ = Model.create_default()
    env = PopDownGym(config, model)
    return env

def sample_random_initial_state(env, prng_key):
    """Sample a random initial state around the nominal initial state."""
    initial_state = jax.tree_util.tree_map(
        lambda x: jnp.array(x), env.nominal_initial_state
    )
    initial_state["Hmode"] = jnp.array(1)
    fractional_variation = jax.tree_util.tree_map(
        lambda x: 0.01 * x * jax.random.uniform(prng_key, minval=-1.0, maxval=1.0),
        env.RANDOM_INITIAL_STATE_PERCENT_VAR,
    )
    for key, variation in fractional_variation.items():
        initial_state[key] = initial_state[key] * (1 + variation)

    return initial_state

def benchmark_step(num_trials: int = 100):
    """Speed benchmarking for stepping the environment."""
    env = create_env()
    env.reset()
    control = env.action_space.sample()
    state = sample_random_initial_state(env, jax.random.PRNGKey(0))

    # Wrap the step function to test
    def step(state, action):
        action = {
            action_name: action[i]
            for i, action_name in enumerate(env.ACTION_RANGES.keys())
        }
        unnormalized_action = env.unnormalize_action(action)

        # Step the environment and compute the reward
        next_state, reward_inputs = env._step(state, unnormalized_action)
        reward, _ = env.reward_model.reward(reward_inputs, unnormalized_action)

        return next_state, reward

    no_jit_times = []
    step_fn = step
    # Burn-in
    step_fn(state, control)
    # Run trials
    for _ in tqdm(range(num_trials)):
        start = time.perf_counter()
        step_fn(state, control)
        no_jit_times.append(time.perf_counter() - start)

    jit_times = []
    step_fn = jax.jit(step)
    # Burn-in
    step_fn(state, control)
    # Run trials
    for _ in tqdm(range(num_trials)):
        start = time.perf_counter()
        step_fn(state, control)
        jit_times.append(time.perf_counter() - start)

    print(f"Benchmarking PopDownGym step and reward with {num_trials} trials")
    print(f"Without jit: {np.mean(no_jit_times)} +/- {np.std(no_jit_times)}")
    print(f"With jit: {np.mean(jit_times)} +/- {np.std(jit_times)}")


def benchmark_simulate(num_trials: int = 100):
    """Speed benchmarking for stepping the environment."""
    env = create_env()
    env.reset()

    # Wrap the step function to test
    def simulate_trajectory(initial_state, actions):
        # Define a function for jax.lax.scan
        def step(carry, input):
            # Unpack the carry
            state = carry
            action = input

            # Map the array of actions to a dict and unnormalize
            action = {
                action_name: action[i]
                for i, action_name in enumerate(env.ACTION_RANGES.keys())
            }
            unnormalized_action = env.unnormalize_action(action)

            # Step the environment and compute the reward
            next_state, reward_inputs = env._step(state, unnormalized_action)
            reward, _ = env.reward_model.reward(reward_inputs, unnormalized_action)

            # prepare the carry for the next iteration
            carry = next_state
            output = (reward, next_state)

            return next_state, output

        # Simulate the trajectory
        _, (rewards, states) = jax.lax.scan(step, initial_state, actions)

        return rewards.sum(), states
    
    # Scan over a range of vmap dimensions
    batch_sizes = [2 ** i for i in range(20)][-2:]
    num_steps = 100
    results = []
    for batch_size in tqdm(batch_sizes):
        # Define constant initial state and controls
        control = env.action_space.sample()
        state = sample_random_initial_state(env, jax.random.PRNGKey(0))
        # Add a time dimension to control
        control = jnp.stack([control] * num_steps)
        # Stack the state and control to run multiple trials in parallel
        state = jax.tree_map(lambda x: jnp.stack([x] * batch_size), state)
        control = jnp.stack([control] * batch_size)

        # Burn-in
        fn = jax.jit(jax.vmap(simulate_trajectory))
        fn(state, control)
        # Run trials
        for _ in range(num_trials):
            start = time.perf_counter()
            fn(state, control)
            eval_time = time.perf_counter() - start
            
            log = {
                "batch_size": batch_size,
                "num_steps": num_steps,
                "num_trials": num_trials,
                "variant": "jit-vmap-scan",
                "eval_time": eval_time,
            }
            results.append(log)

    # results = pd.DataFrame(results)
    # results.to_csv("benchmarking/gpu_benchmark_simulate.csv", index=False)


if __name__ == "__main__":
    benchmark_simulate(num_trials=100)

    # # Load data from csv
    # results = pd.read_csv("benchmarking/gpu_benchmark_simulate.csv")

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # sns.set_style("whitegrid")

    # # Add a column to the data to be evaluation time per sample
    # results["Time per trajectory (s)"] = results.eval_time / results.batch_size

    # # Plot the results of evaluation time per sample vs batch size
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # sns.lineplot(
    #     data=results,
    #     x="batch_size",
    #     y="eval_time",
    #     hue="variant",
    #     ax=ax,
    # )
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylim(top=100)

    # # Save the figure
    # fig.savefig("benchmarking/gpu_benchmark_simulate.png", dpi=300)