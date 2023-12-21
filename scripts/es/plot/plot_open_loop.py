"""
Plot the open-loop (feedforward controls) trajectories
"""
import os

import equinox as eqx
import jax
import matplotlib.pyplot as plt

from pop_down_gym.pd_gym_stateless import PopDownGymStateless
from scripts.es.train.es_closed_loop import plot_test_set_trajectories
from scripts.es.train.es_open_loop import CubicTrajectory, rollout_open_loop

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
    results_dir = (
        "tmp/es/open_loop_sparse-noterminate_cubic/uncertainty_1.00/lr_1.0e-01"
    )
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
    ) = jax.vmap(rollout_test, in_axes=(0, None))(keys, best_trajectory)

    # Plot
    plt.style.use("ggplot")
    plot_test_set_trajectories(
        env,
        t_test,
        reward_inputs_test,
        results_dir,
        use_wandb=False,
    )
