"""
Plot the open-loop (feedforward controls) trajectories
"""
import os

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils.plot_utils import setup_nature_style
from pop_down_gym.pd_gym_stateless import PopDownGymStateless
from scripts.es.train.es_closed_loop import plot_test_set_trajectories
from scripts.es.train.es_open_loop import CubicTrajectory, rollout_open_loop


def max_constraint_violation(reward_inputs, upper_bounds):
    """Compute the maximum constraint violation."""
    total_violation = 0.0
    for key in upper_bounds:
        violation = (reward_inputs[key] >= upper_bounds[key]) * env.reward_model.params[
            "hit_barrier_reward"
        ]
        total_violation += violation.mean()

    return total_violation


def plot_parameters_vs_constraint_violation(env, reward_inputs, params, save_dir):
    """Plot the parameters against the constraint violation."""
    violation = jax.vmap(max_constraint_violation, in_axes=(0, None))(
        reward_inputs, env.reward_model.limits
    )

    # Re-map param keys to latex
    new_param_keys = {
        "ion_dilution": r"$k_{dil}$",
        "hl_factor": r"$k_{HL}$",
        "Hfactor": r"$H$",
        "Zeff": r"$Z_{eff}$",
        "Te_over_Ti": r"$k_{te\_ti}$",
        "tau_n_factor": r"$k_{N}$",
        "prad_mult": r"$k_{rad}$",
    }
    params = {new_param_keys[key]: value for key, value in params.items()}

    # Normalize values by their min/max range
    param_ranges = {
        new_param_keys[key]: (lb, ub - lb)
        for key, (lb, ub) in env.random_param_ranges.items()
    }
    normalized_params = {
        key: (value - param_ranges[key][0]) / param_ranges[key][1]
        for key, value in params.items()
    }

    df = pd.DataFrame(
        {key: value for key, value in normalized_params.items()}
        | {
            "Max. constraint violation": violation,
        }
    )
    params_of_interest = [
        r"$k_{dil}$",
        r"$k_{HL}$",
        r"$H$",
        r"$k_{te\_ti}$",
    ]
    g = sns.PairGrid(df, hue="Max. constraint violation", vars=params_of_interest)
    g.map(sns.scatterplot, size=2)

    # Save
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "params_vs_violation.png"))

    plt.close()


if __name__ == "__main__":
    # Hyperparams
    num_control_points = 10
    num_eval_rollouts = 100
    simulation_steps = 100

    # Set the seed for reproducibility
    prng_key = jax.random.PRNGKey(0)

    # Load the environment
    env = PopDownGymStateless.create_env()

    # Load the trajectory
    prng_key, subkey = jax.random.split(prng_key)
    initial_traj = CubicTrajectory(subkey, num_control_points, env.n_actions)
    results_dir = "tmp/es/open_loop_dense-cubic-tuned/uncertainty_1.00/lr_1.0e-01"
    best_trajectory = eqx.tree_deserialise_leaves(
        os.path.join(results_dir, "best_trajectory.eqx"), initial_traj
    )

    # Simulate a bunch of trajectories
    keys = jax.random.split(prng_key, num_eval_rollouts)
    rollout_test = lambda key, traj: rollout_open_loop(key, env, traj, simulation_steps)  # noqa
    (
        rewards_test,
        states_test,
        t_test,
        reward_inputs_test,
        _,
        hit_goal_test,
        actions_test,
        params_test,
    ) = jax.vmap(rollout_test, in_axes=(0, None))(keys, best_trajectory)

    # Plot
    plt.style.use("ggplot")
    setup_nature_style()
    plot_test_set_trajectories(
        env,
        t_test,
        reward_inputs_test,
        results_dir,
        use_wandb=False,
    )

    plot_parameters_vs_constraint_violation(
        env, reward_inputs_test, params_test, results_dir
    )
