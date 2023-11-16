"""Tests and benchmarking for PopDownGymStateless"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from pop_down_gym.model import Model
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def create_env():
    """Create an instance of PopDownGymStateless."""
    config_filepath = os.path.join(os.path.dirname(__file__), "../configs/gym.yaml")
    config = yaml.safe_load(open(config_filepath, "r"))
    model, _ = Model.create_default()
    env = PopDownGymStateless(config, model)
    return env


def test_env_sample_state():
    # Create an environment
    env = create_env()

    # Sample a random state
    key = jax.random.PRNGKey(0)
    state = env.sample_state(key)

    assert state is not None
    assert isinstance(state, dict)


def test_env_sample_params():
    # Create an environment
    env = create_env()

    # Sample a random state
    key = jax.random.PRNGKey(0)
    params = env.sample_params(key)

    assert params is not None
    assert isinstance(params, dict)


def benchmark_simulate(num_trials: int = 100):
    """Speed benchmarking for stepping the environment."""
    env = create_env()
    prng_key = jax.random.PRNGKey(0)

    # Scan over a range of vmap dimensions
    batch_sizes = [2**i for i in range(20)]
    num_steps = 100
    results = []
    for batch_size in tqdm(batch_sizes):
        # Define random controls
        prng_key, action_key = jax.random.split(prng_key)
        action_keys = jax.random.split(action_key, num_steps)
        actions = jax.vmap(env.sample_action)(action_keys)

        # Stack actions across a batch dimension
        actions = jnp.stack([actions] * batch_size)
        sim_keys = jax.random.split(prng_key, batch_size)

        # Burn-in
        fn = lambda key, actions: env.simulate_trajectory_open_loop(
            key, actions, num_steps
        )
        fn = jax.jit(jax.vmap(fn))
        fn(sim_keys, actions)

        # Run trials
        for _ in range(num_trials):
            start = time.perf_counter()
            fn(sim_keys, actions)
            eval_time = time.perf_counter() - start

            log = {
                "batch_size": batch_size,
                "num_steps": num_steps,
                "num_trials": num_trials,
                "variant": "jit-vmap-scan",
                "eval_time": eval_time,
            }
            results.append(log)

    results = pd.DataFrame(results)
    results.to_csv("benchmarking/gpu_benchmark_simulate_stateless.csv", index=False)


if __name__ == "__main__":
    benchmark_simulate(num_trials=1)

    # Load data from csv
    results = pd.read_csv("benchmarking/gpu_benchmark_simulate_stateless.csv")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # Add a column to the data to be evaluation time per sample
    results["Time per trajectory (s)"] = results.eval_time / results.batch_size

    # Plot the results of evaluation time per sample vs batch size
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.lineplot(
        data=results,
        x="batch_size",
        y="eval_time",
        hue="variant",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=100)

    # Save the figure
    fig.savefig("benchmarking/gpu_benchmark_simulate_stateless.png", dpi=300)
