import time

import jax
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm
from typing import List

from pop_down_gym.pd_gym_stateless import PopDownGymStateless

def benchmark_simulate(batch_sizes: List[int], num_trials: int = 100):
    """Speed benchmarking for stepping the environment."""
    env = PopDownGymStateless.create_env()
    prng_key = jax.random.PRNGKey(0)

    # Scan over a range of vmap dimensions
    num_steps = 100
    results = []
    results_burn_in = []
    for batch_size in tqdm(batch_sizes):
        # Define random controls
        prng_key, action_key = jax.random.split(prng_key)
        action_keys = jax.random.split(action_key, num_steps)
        actions = jax.vmap(env.sample_action)(action_keys)

        # Stack actions across a batch dimension
        actions = jnp.stack([actions] * batch_size)
        sim_keys = jax.random.split(prng_key, batch_size)

        # Burn-in
        def fn(key, action):
            return env.simulate_trajectory_open_loop(key, action, num_steps)
        
        start = time.perf_counter()
        fn = jax.jit(jax.vmap(fn))
        jax.block_until_ready(fn(sim_keys, actions))
        burn_in_time = time.perf_counter() - start
        results_burn_in.append(
            {
                "batch_size": batch_size,
                "num_steps": num_steps,
                "eval_time": burn_in_time,
            }
        )

        # Run trials
        for _ in range(num_trials):
            start = time.perf_counter()
            jax.block_until_ready(fn(sim_keys, actions))
            eval_time = time.perf_counter() - start

            log = {
                "batch_size": batch_size,
                "num_steps": num_steps,
                "num_trials": num_trials,
                "eval_time": eval_time,
            }
            results.append(log)

    results = pd.DataFrame(results)
    results_burn_in = pd.DataFrame(results_burn_in)
    return results, results_burn_in

if __name__ == "__main__":
    batch_sizes = [2**i for i in range(20)]
    results, results_burn_in = benchmark_simulate(batch_sizes, num_trials=10)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 14})

    # Add a column to the data to be evaluation time per sample
    results["Time per trajectory (s)"] = results.eval_time / results.batch_size

    # Plot the results of evaluation time per sample vs batch size
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.loglog(
        results.batch_size,
        results.eval_time,
        label="After JIT",
    )
    ax.loglog(
        results_burn_in.batch_size,
        results_burn_in.eval_time,
        label="JIT",
    )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Wall-clock Time (s)")
    ax.set_ylim(top=100)
    ax.set_title("Benchmarking PopDownGym")
    ax.legend()
    plt.grid(True, which="both", ls="-")
    
    # Save the figure
    fig.savefig("gpu_benchmark_simulate_stateless.png", dpi=300)