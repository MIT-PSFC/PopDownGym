import os

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
from es.train.es_open_loop import NUM_CONTROL_POINTS, CubicTrajectory

import pop_down_gym
from pop_down_gym.pd_gym_jaxrl import default_build_ppo
from pop_down_gym.pd_gym_stateless import PopDownGymStateless
from pop_down_gym.raptor.raptor_rd_gym import RaptorRDGym
from pop_down_gym.raptor.utils import convert_to_df


class PolicyInterface:
    def __init__(self, model_path, model_type):
        self.model_type = model_type
        if model_type == "PPO_OSO" or model_type == "BASELINE":
            shift_dict = {
                "beta_n": -1.0,
                "li": 0.0,
                "beta_p": -0.6,
                "ng_frac": -0.9,
            }
            ppo, env, offset_dict, tmp_dir = default_build_ppo(shift_dict)
            self.train_env = env.pd
            self.constraint_limits = self.train_env.reward_model.limits
        else:
            self.train_env = PopDownGymStateless.create_env()
            self.constraint_limits = self.train_env.reward_model.limits
        if model_type == "PPO_OSO":

            def model_fn(obs, t=None):
                obs = self.train_env.flatten_obs(obs)
                constraint_shifts = jnp.zeros(8)
                for key, val in shift_dict.items():
                    # Get index of this key in the environment.
                    key_list = list(self.constraint_limits.keys())
                    idx = key_list.index(key)

                    # Set the constraint shift in the policy input.
                    constraint_shifts = constraint_shifts.at[idx].set(val)
                action = ppo.act(jnp.concatenate([obs, constraint_shifts]))
                unnormalized_action = self.train_env.dictify_and_unnormalize_action(
                    action
                )
                return unnormalized_action

            self.model_fn = model_fn
        elif model_type == "BASELINE":

            def model_fn(obs, t=None):
                unnormalized_action = {
                    "dIp_dt": -1.45,
                    "dPaux_dt": -1,
                    "fueling19": 4.0,
                    "dgs_dt": 0.225,
                }
                return unnormalized_action

            self.model_fn = model_fn
        elif model_type == "ES_OPENLOOP_DAWSON":
            traj = CubicTrajectory(
                jax.random.PRNGKey(0),
                NUM_CONTROL_POINTS,
                self.train_env.n_actions,
                self.train_env.time_limit,
            )
            traj = eqx.tree_deserialise_leaves(
                os.path.join(
                    model_path, "uncertainty_1.00/lr_1.0e-01/best_trajectory.eqx"
                ),
                traj,
            )
            self.model_fn = (
                lambda obs, t: self.train_env.dictify_and_unnormalize_action(
                    traj(jnp.array(t))
                )
            )
        else:
            raise ValueError("model_type must be one of PPO_SB3, ES_DAWSON, PPO_OSO")

    def state_to_action(self, train_env_state, t):
        obs = self.train_env.state_to_obs(train_env_state)
        action = self.model_fn(obs, t)
        return action

    def step_train_env(self, action):
        return self.train_env.step(action)


def tree_transpose_to_arrays(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of arrays."""
    # Concatenate along the first axis.
    # Note we use concatenate instead of stack because stack can create new axes.
    return jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)


def run_with_raptor_loop(
    raptor_gym, policy_interface, max_steps=140, smooth_length: int = 5
):
    logged_actions = []
    action_buffer = []
    for i in range(max_steps):
        obs_for_pd_gym = raptor_gym.obs_for_pd_gym()
        # We have reached the goal.
        if obs_for_pd_gym["Ip_MA"] < 2.0:
            break
        unnormalized_action = policy_interface.state_to_action(
            obs_for_pd_gym, t=raptor_gym.time
        )

        raptor_action_input = {
            "dIp_dt": 1e6 * unnormalized_action["dIp_dt"],
            "dPaux_dt": 1e6 * unnormalized_action["dPaux_dt"],
            "fueling19": unnormalized_action["fueling19"],
            "dgs_dt": unnormalized_action["dgs_dt"],
        }

        # Perform smoothing of this action with past ones.
        action_buffer.append(raptor_action_input)
        action_buffer_tree_transposed = tree_transpose_to_arrays(
            action_buffer[-smooth_length:]
        )

        # Execute the smoothed action.
        raptor_action_input_smoothed = jax.tree_map(
            lambda x: jnp.mean(x), action_buffer_tree_transposed
        )
        try:
            raptor_gym.step(raptor_action_input_smoothed)
            logged_actions.append(
                {
                    "time": raptor_gym.time,
                    "gs": obs_for_pd_gym["gs"],
                    "Paux": obs_for_pd_gym["Paux"],
                    **raptor_action_input_smoothed,
                }
            )
        except:
            print("Raptor step failed, exiting.")
            break
    raptor_out = raptor_gym.raptor_out()
    return raptor_out, logged_actions


@click.command()
@click.option(
    "--model_path",
    type=str,
    default=os.path.join(
        os.path.dirname(__file__), "../../tmp/ppo_adj_ckpt/checkpoint"
    ),
)
@click.option("--model_type", type=str, default="PPO_OSO")
@click.option("--raptor_path", type=str, default="/home/awang/raptor")
def main(model_path, model_type, raptor_path):
    policy_interface = PolicyInterface(model_path=model_path, model_type=model_type)
    gym_dt = policy_interface.train_env.dt
    raptor_dt = 0.005

    raptor_steps_per_gym_step = gym_dt / raptor_dt
    # Assert near int.
    assert abs(raptor_steps_per_gym_step - round(raptor_steps_per_gym_step)) < 1e-6
    raptor_steps_per_gym_step = int(round(raptor_steps_per_gym_step))

    raptor_gym = RaptorRDGym(raptor_path, raptor_dt, raptor_steps_per_gym_step)
    raptor_gym.reset()
    assert raptor_dt * raptor_steps_per_gym_step == policy_interface.train_env.dt
    raptor_out, actions = run_with_raptor_loop(raptor_gym, policy_interface)
    raptor_out_df = convert_to_df(raptor_out)
    df = pd.DataFrame(actions)
    df = df.set_index("time")
    df = df.join(raptor_out_df, how="outer")
    df.attrs["constraint_limits"] = policy_interface.constraint_limits

    df.to_pickle(
        os.path.join(pop_down_gym.ROOT_DIR, f"../tmp/sim2sim_{model_type}.pkl")
    )


if __name__ == "__main__":
    main()
