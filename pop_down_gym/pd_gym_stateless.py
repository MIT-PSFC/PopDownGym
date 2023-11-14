import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import pop_down_gym.physics as physics
from contrax.simulate import SimFFControl
from pop_down_gym.model import Model
from pop_down_gym.reward import RewardModel
from pop_down_gym.utils import remap_range


class PopDownGymStateless:
    CONT_STATE_RANGES = {
        "li": (0.5, 6.0),  # Normalized internal inductance [-]
        "Ip_MA": (1.0, 9.0),  # Plasma current [MA]
        "vc_minus_vb": (-5.0, 5.0),  # vc-vb as defined by Romero [V]
        "Wth": (1e5, 3e7),  # Stored thermal energy [J]
        "nfuel19_vol": (1.0, 30.0),  # Volume averaged fuel density [10^19 m^-3]
        "Paux": (0.0, 25.0),  # Auxiliary heating power [MW]
        "gs": (0.0, 1.0),  # Geometry evolution parameter [0]
    }

    """
    When generating a random initial state, we allow each state variable
    to vary by a certain percentage of its nominal value. If the variable
    is not included in this dict, then it is not varied.
    """
    RANDOM_INITIAL_STATE_PERCENT_VAR = {
        "li": 10.0,
        "Ip_MA": 2.0,
        "vc_minus_vb": 10.0,
        "Wth": 2.0,
        "nfuel19_vol": 2.0,
    }
    ACTION_RANGES = {
        "dIp_dt": (-3.0, -0.5),  # Rate of change of plasma current [MA/s]
        "dPaux_dt": (-5.0, 5.0),  # Rate of change of auxiliary power [MW/s]
        "fueling19": (
            0.0,
            10.0,
        ),  # Really limit fueling, prob don't want much [10^19/s]
        "dgs_dt": (0.0, 1.0),  # Rate of evolution through geometry space [1/s]
    }

    DT = 0.05

    RANDOM_PARAM_RANGES = {
        "ion_dilution": (0.8, 0.9),
        "HL_FUDGE": (0.55, 0.75),
        "Hfactor": (0.8, 1.0),
        "Zeff": (1.2, 1.8),
        "Te_over_Ti": (1.0, 1.2),
        "tau_n_factor": (7.0, 9.0),
        "prad_mult": (2.0, 3.0),
    }
    TIME_LIMIT = 7.5

    def __init__(self, cfg: dict, model: Model):
        self.simulator = SimFFControl(model, dt0=1e-2)
        self.reward_model = RewardModel(cfg["reward"])
        self.shot_constants = model.shot_constants
        self.nominal_initial_state = cfg["nominal_initial_state"]
        # Declare the normalized action space.
        self.action_space = jnp.vstack(
            (-1.0 * np.ones(len(self.ACTION_RANGES)), np.ones(len(self.ACTION_RANGES)))
        )

    def sample_state(self, prng_key):
        """
        Generate a random initial state.

        Args:
            prng_key (jax.random.PRNGKey): A PRNG key.
        """
        initial_state = jax.tree_util.tree_map(
            lambda x: jnp.array(x), self.nominal_initial_state
        )
        initial_state["Hmode"] = jnp.array(1)
        fractional_variation = jax.tree_util.tree_map(
            lambda x: 0.01 * x * jax.random.uniform(prng_key, minval=-1.0, maxval=1.0),
            self.RANDOM_INITIAL_STATE_PERCENT_VAR,
        )
        for key, variation in fractional_variation.items():
            initial_state[key] = initial_state[key] * (1 + variation)

        return initial_state

    def sample_params(self, prng_key):
        """
        Generate a random set of parameters.

        Args:
            prng_key (jax.random.PRNGKey): A PRNG key.
        """
        # Choose a random parameter uniformly in the range
        params = {}
        for param_key, (lb, ub) in self.RANDOM_PARAM_RANGES.items():
            prng_key, key = jax.random.split(prng_key)
            params[param_key] = jax.random.uniform(key, minval=lb, maxval=ub)

        return params

    def sample_action(self, prng_key):
        return jax.random.uniform(
            prng_key, minval=-1.0, maxval=1.0, shape=(len(self.ACTION_RANGES),)
        )

    @property
    def n_obs(self):
        return len(self.CONT_STATE_RANGES)

    @property
    def n_actions(self):
        return len(self.ACTION_RANGES)

    def reset(self, prng_key):
        """
        Reset the environment.

        Args:
            prng_key (jax.random.PRNGKey): A PRNG key.
        """
        # Compute a random initial state.
        prng_key, state_key = jax.random.split(prng_key)
        state = self.sample_state(state_key)

        # Compute a random parameter set.
        prng_key, param_key = jax.random.split(prng_key)
        random_params = self.sample_params(param_key)
        random_params = jax.tree_map(lambda val: val.squeeze(), random_params)

        info = {"time": 0.0}

        return random_params, state, self.state_to_obs(state), info

    def unnormalize_action(self, action) -> dict:
        """Given an action sampled from the normalized action space, unnormalize it.

        Args:
            action (dict): _description_

        Returns:
            dict: _description_
        """
        for i, (action_name, action_val) in enumerate(action.items()):
            action_space_range = (
                self.action_space[0][i],
                self.action_space[1][i],
            )
            action[action_name] = remap_range(
                action_val, action_space_range, self.ACTION_RANGES[action_name]
            )
        return action

    def check_out_of_bounds(self, obs):
        """Check if the given observations are in bounds or not"""
        # TODO currently, the observation range seems to be [-1, 1], not
        # the range given in CONT_STATE_RANGES
        each_obs_in_bounds = jax.tree_map(
            lambda x: jnp.logical_and(x >= -1.0, x <= 1.0), obs
        )
        out_of_bounds = jnp.logical_not(jnp.all(each_obs_in_bounds))
        return out_of_bounds

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, params, state, action):
        """
        Step the environment forward in time.

        Args:
            params (dict): The current random parameters.
            state (dict): The current state.
            action (dict): The current action.
        """
        ts = jnp.array([0.0, self.DT])  # Time steps.

        # Initial State.
        initial_state = {
            "li": state["li"],
            "Ip_MA": state["Ip_MA"],
            "vc_minus_vb": state["vc_minus_vb"],
            "Wth": state["Wth"],
            "nfuel19_vol": state["nfuel19_vol"],
            "Paux": state["Paux"],
            "gs": state["gs"],
        }
        # [Hmode, Hfactor, Zeff, ion_dilution, Te_over_Ti, f_dt, tau_n_factor, prad_mult]
        params = {
            "Hmode": state["Hmode"],
            "HL_FUDGE": params["HL_FUDGE"],
            "Hfactor": params["Hfactor"],
            "Zeff": params["Zeff"],
            "ion_dilution": params["ion_dilution"],
            "Te_over_Ti": params["Te_over_Ti"],
            "f_dt": 0.5,
            "tau_n_factor": params["tau_n_factor"],
            "prad_mult": params["prad_mult"],
        }

        # SimFFControl needs controls at all time steps in "ts".
        # Let's just assume a zero-order-hold, so constant action over the simulation step.
        controls = jax.tree_map(lambda x: jnp.repeat(x, 2), action)
        res = self.simulator.simulate(ts, initial_state, controls, params=params)

        # Evaluate the model at the first and last time steps in debug mode to get info.
        ys0 = jax.tree_map(lambda x: x[0], res.ys)
        controls0 = jax.tree_map(lambda x: x[0], controls)
        ys_last = jax.tree_map(lambda x: x[-1], res.ys)
        controls_last = jax.tree_map(lambda x: x[-1], controls)
        _, info0 = self.simulator.model(ys0, controls0, params, debug=True)
        _, info = self.simulator.model(ys_last, controls_last, params, debug=True)

        """
        Determine if the HL transition occured.
        """
        Ploss = jnp.abs(info["Ploss"])
        PLH = physics.PLH_threshold(
            info["ne19_line"] / 10,  # Convert to ne20.
            self.shot_constants.Bphi0,
            info["aminor"],
            self.shot_constants.R0,
        )
        PHL = params["HL_FUDGE"] * PLH
        Hmode_new = jnp.where(
            jnp.logical_and(state["Hmode"] == 1, Ploss < PHL), 0, state["Hmode"]
        )

        state_new = {
            "li": ys_last["li"],
            "Ip_MA": ys_last["Ip_MA"],
            "vc_minus_vb": ys_last["vc_minus_vb"],
            "Wth": ys_last["Wth"],
            "nfuel19_vol": ys_last["nfuel19_vol"],
            "Paux": jnp.clip(
                ys_last["Paux"],
                self.CONT_STATE_RANGES["Paux"][0],
                self.CONT_STATE_RANGES["Paux"][1],
            ),
            "gs": jnp.clip(
                ys_last["gs"],
                self.CONT_STATE_RANGES["gs"][0],
                self.CONT_STATE_RANGES["gs"][1],
            ),
            "Hmode": Hmode_new,
        }

        beta_p = physics.pressure_to_beta_p(
            info["pressure_vol_avg"],
            state_new["Ip_MA"],
            info["aminor"],
        )
        beta_t = physics.pressure_to_beta(
            info["pressure_vol_avg"], self.shot_constants.Bphi0
        )
        beta_n = physics.betas_to_beta_n(
            beta_p,
            beta_t,
            state_new["Ip_MA"],
            info["aminor"],
            self.shot_constants.Bphi0,
        )

        ng_frac = physics.greenwald_fraction(
            info["ne19_line"],
            state_new["Ip_MA"],
            info["aminor"],
        )

        beta_p0 = physics.pressure_to_beta_p(
            info0["pressure_vol_avg"],
            state["Ip_MA"],
            info0["aminor"],
        )
        Bv0 = physics.calc_Bv(
            state["Ip_MA"],
            info0["kappa_a"],
            beta_p=beta_p0,
            li3=state["li"],  # close enough
            R=self.shot_constants.R0,
            a=info0["aminor"],
        )

        Bv = physics.calc_Bv(
            state_new["Ip_MA"],
            info["kappa_a"],
            beta_p=beta_p,
            li3=state["li"],  # close enough
            R=self.shot_constants.R0,
            a=info["aminor"],
        )

        reward_inputs = {
            "li": state_new["li"],
            "Ip_MA": state_new["Ip_MA"],
            "kappa": info["kappa_a"],
            "beta_p": beta_p,
            "beta_t": beta_t,
            "beta_n": beta_n,
            "ng_frac": ng_frac,
            "Wdot_mag": jnp.abs(info["Wdot"]),
            "Bv_dot_mag": jnp.abs((Bv - Bv0) / self.DT),
        }
        return state_new, reward_inputs

    def step(self, t, params, state, action):
        """
        Step the environment forward in time.
        """
        # Map the array of actions to a dict.
        action = {
            action_name: action[i]
            for i, action_name in enumerate(self.ACTION_RANGES.keys())
        }
        unnormalized_action = self.unnormalize_action(action)
        new_state, reward_inputs = self._step(params, state, unnormalized_action)
        reward, reward_terms = self.reward_model.reward(
            reward_inputs, unnormalized_action
        )
        next_time = t + self.DT

        truncated = next_time >= self.TIME_LIMIT
        hit_goal = reward_terms["hit_goal_reward"] > 0.0
        obs = self.state_to_obs(new_state)
        out_of_bounds = self.check_out_of_bounds(obs)
        terminated = jnp.logical_or(truncated, out_of_bounds)
        terminated = jnp.logical_or(terminated, hit_goal)

        info = {
            "time": next_time,
            "state": new_state,
            "action": unnormalized_action,
            "reward_inputs": reward_inputs,
            "reward_terms": reward_terms,
            "random_params": params,
            "hit_goal": hit_goal,
            "out_of_bounds": out_of_bounds,
        }

        return obs, reward, terminated, truncated, info

    def state_to_obs(self, state):
        continuous = {k: v for k, v in state.items() if k != "Hmode"}
        obs = jnp.zeros(len(continuous))
        # In this problem, we define the observations as the continuous states normalized to [-1, 1].
        # TODO(allenw): fair question to be asked
        for i, (key, value) in enumerate(continuous.items()):
            obs = obs.at[i].set(
                remap_range(value, self.CONT_STATE_RANGES[key], (-1.0, 1.0))
            )
        return obs

    def simulate_trajectory_open_loop(self, prng_key, open_loop_actions, steps=100):
        """
        Simulate an open-loop trajectory for a fixed number of steps.

        Starts from a random initial state and with random parameters

        Args:
            prng_key (jax.random.PRNGKey): A PRNG key.
            open_loop_controls (dict): A dictionary of open-loop controls.
            steps (int): The number of steps to simulate.
        """
        # Sample random parameters and initial state
        state_key, param_key = jax.random.split(prng_key)
        initial_state = self.sample_state(state_key)
        params = self.sample_params(param_key)

        # Define a step function to simulate using scan
        def scan_step(carry, input):
            # Unpack the carry
            state, t = carry
            action = input

            # Step the environment
            _, reward, _, _, info = self.step(t, params, state, action)

            # prepare the carry for the next iteration
            carry = (info["state"], info["time"])
            output = (reward, info["state"])

            return carry, output

        # Simulate the trajectory
        _, (rewards, states) = jax.lax.scan(
            scan_step, (initial_state, 0.0), open_loop_actions
        )

        return rewards.sum(), states


if __name__ == "__main__":
    config_filepath = os.path.join(os.path.dirname(__file__), "configs/gym.yaml")
    config = yaml.safe_load(open(config_filepath, "r"))
    model, _ = Model.create_default()
    env = PopDownGymStateless(config, model)
    key = jax.random.PRNGKey(0)
    params, state, obs, info = env.reset(key)
    obs, reward, terminated, truncated, info = env.step(
        info["time"], params, state, env.sample_action(key)
    )
