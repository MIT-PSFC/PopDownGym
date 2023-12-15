import os
from functools import partial
from typing import Tuple, Dict
from jaxtyping import PyTree

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import pop_down_gym.physics as physics
from contrax.simulate import SimFFControl
import pop_down_gym
from pop_down_gym.model import Model
from pop_down_gym.reward import RewardModel
from pop_down_gym.utils import remap_range


class PopDownGymStateless:
    CONT_STATES = ["li", "Ip_MA", "vc_minus_vb", "Wth", "nfuel19_vol", "Paux", "gs"]

    def __init__(self, cfg: dict, model: Model):
        """
        Unpack the configuration file.
        """
        self.cont_state_ranges = {
            key: tuple(val) for key, val in cfg["cont_state_ranges"].items()
        }
        self.random_initial_state_percent_var = cfg["random_initial_state_percent_var"]
        self.action_ranges = {
            key: tuple(val) for key, val in cfg["action_ranges"].items()
        }
        self.random_param_ranges = {
            key: tuple(val) for key, val in cfg["random_param_ranges"].items()
        }
        self.dt = cfg["dt"]
        self.time_limit = cfg["time_limit"]

        # Initialize the simulator.
        self.simulator = SimFFControl(
            model, dt0=cfg["dt"] / 5.0
        )  # Do 5 steps per gym dt.
        self.reward_model = RewardModel(cfg["reward"])
        self.shot_constants = model.shot_constants
        self.nominal_initial_state = cfg["nominal_initial_state"]

        # Declare the normalized action space.
        self.action_space = jnp.vstack(
            (-1.0 * np.ones(len(self.action_ranges)), np.ones(len(self.action_ranges)))
        )

        print(repr(list(self.action_ranges.keys())))
        exit(0)

        # Declare the normalized observation space.
        self.observation_space = {
            "continuous": jnp.vstack(
                (
                    -1.0 * np.ones(len(self.cont_state_ranges)),
                    np.ones(len(self.cont_state_ranges)),
                )
            ),
            "Hmode": (0, 1),
        }

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
            self.random_initial_state_percent_var,
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
        for param_key, (lb, ub) in self.random_param_ranges.items():
            prng_key, key = jax.random.split(prng_key)
            params[param_key] = jax.random.uniform(key, minval=lb, maxval=ub)

        return params

    def sample_action(self, prng_key):
        return jax.random.uniform(
            prng_key, minval=-1.0, maxval=1.0, shape=(len(self.action_ranges),)
        )

    @property
    def n_obs(self):
        # The number of observations is the number of continuous states.
        # Plus the H-Mode observation.
        cont_states = len(self.cont_state_ranges)
        non_cont_obs = {k: v for k, v in self.observation_space.items() if k != "continuous"}
        return cont_states + len(non_cont_obs)

    @property
    def n_actions(self):
        return len(self.action_ranges)

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

    def dictify_and_unnormalize_action(self, action) -> dict:
        """Given an action sampled from the normalized action space, unnormalize it.

        Args:
            action (dict): _description_

        Returns:
            dict: _description_
        """
        action = {
            action_name: action[i]
            for i, action_name in enumerate(self.action_ranges.keys())
        }
        for i, (action_name, action_val) in enumerate(action.items()):
            action_space_range = (
                self.action_space[0][i],
                self.action_space[1][i],
            )
            action[action_name] = remap_range(
                action_val, action_space_range, self.action_ranges[action_name]
            )
        return action

    def check_out_of_bounds(self, obs):
        """Check if the given observations are in bounds or not"""

        def in_bound(i):
            return jnp.logical_and(
                obs["continuous"][i] >= self.observation_space["continuous"][0][i],
                obs["continuous"][i] <= self.observation_space["continuous"][1][i],
            )

        continuous_obs_in_bounds = jax.tree_map(
            in_bound, jnp.arange(self.observation_space["continuous"].shape[1])
        )
        hmode_in_bounds = jnp.logical_or(
            obs["Hmode"] == self.observation_space["Hmode"][0],
            obs["Hmode"] == self.observation_space["Hmode"][1],
        )

        each_obs_in_bounds = jnp.logical_and(
            jnp.all(continuous_obs_in_bounds), hmode_in_bounds
        )

        out_of_bounds = jnp.logical_not(jnp.all(each_obs_in_bounds))
        return out_of_bounds

    @partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: PyTree[float], action: PyTree[float], random_params: PyTree[float]
    ) -> Tuple[PyTree[float], PyTree[float]]:
        """Simulate the dynamics by one step.

        Args:
            state (PyTree[float]): state.
            action (PyTree[float]): action.
            random_params (PyTree[float]): random parameters.

        Returns:
            Tuple[PyTree[float], PyTree[float]]: _description_
        """
        ts = jnp.array([0.0, self.dt])  # Time steps.

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
            "Hfactor": random_params["Hfactor"],
            "Zeff": random_params["Zeff"],
            "ion_dilution": random_params["ion_dilution"],
            "Te_over_Ti": random_params["Te_over_Ti"],
            "f_dt": 0.5,
            "tau_n_factor": random_params["tau_n_factor"],
            "prad_mult": random_params["prad_mult"],
        }

        # SimFFControl needs controls at all time steps in "ts".
        # Let's just assume a zero-order-hold, so constant action over the simulation step.
        controls = jax.tree_map(lambda x: jnp.repeat(x, 2), action)
        res = self.simulator.simulate(ts, initial_state, controls, params)

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

        # While there are well-established L->H transition thresholds, the H->L transition is less well understood.
        # It is often assumed that the H->L transition occurs at some fraction of the L->H threshold.
        PHL = random_params["hl_factor"] * PLH
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
                self.cont_state_ranges["Paux"][0],
                self.cont_state_ranges["Paux"][1],
            ),
            "gs": jnp.clip(
                ys_last["gs"],
                self.cont_state_ranges["gs"][0],
                self.cont_state_ranges["gs"][1],
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

        shafranov_coeff0 = physics.shafranov_coeff(
            R0=self.shot_constants.R0,
            aminor=info0["aminor"],
            kappa=info0["kappa"],
            beta_p=beta_p0,
            li3=state[
                "li"
            ],  # li = li3 under the assumption that the plasma is perfectly toroidal.
        )

        shafranov_coeff = physics.shafranov_coeff(
            R0=self.shot_constants.R0,
            aminor=info["aminor"],
            kappa=info["kappa"],
            beta_p=beta_p,
            li3=state_new[
                "li"
            ],  # li = li3 under the assumption that the plasma is perfectly toroidal.
        )

        Bv0 = physics.calc_Bv(
            state["Ip_MA"],
            R0=self.shot_constants.R0,
            shafranov_coeff=shafranov_coeff0,
        )

        Bv = physics.calc_Bv(
            state_new["Ip_MA"],
            R0=self.shot_constants.R0,
            shafranov_coeff=shafranov_coeff,
        )

        q95 = physics.q95(
            Ip_MA=state_new["Ip_MA"],
            B0=self.shot_constants.Bphi0,
            R0=self.shot_constants.R0,
            aminor=info["aminor"],
            kappa=info["kappa"],
            # Note: delta is not really a constant, but RAPTOR
            # isn't computing it, so use the flattop value
            # as a constant.
            delta=self.shot_constants.delta,
            # We don't seem to have data on this squareness
            # factor for SPARC. According to Sauter, a value
            # of 1.0 corresponds to zero squareness so we'll
            # just use that.
            w07=1.0,
        )

        # iota = 1.0/q by definition.
        # Use iota95 as our reward formulation limits growth.
        # If we want q95 > x, then we say iota < 1/x.
        iota95 = 1.0 / q95

        reward_inputs = {
            "li": state_new["li"],
            "Ip_MA": state_new["Ip_MA"],
            "kappa_a": info["kappa_a"],
            "beta_p": beta_p,
            "beta_t": beta_t,
            "beta_n": beta_n,
            "ng_frac": ng_frac,
            "Wdot_mag": jnp.abs(info["Wdot"]),
            "Bv_dot_mag": jnp.abs((Bv - Bv0) / self.dt),
            "shafranov_coeff": shafranov_coeff,
            "iota95": iota95,
        }
        return state_new, reward_inputs

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        t: float,
        params: PyTree[float],
        state: PyTree[float],
        action: PyTree[float],
    ) -> Tuple[PyTree[float], float, bool, bool, Dict]:
        """Step the environment forward in time. Really just a wrapper for the business logic of the environment.

        Args:
            t (float): time (s)
            params (PyTree[float]): external parameters.
            state (PyTree[float]): system state.
            action (PyTree[float]): action.

        Returns:
            Tuple[PyTree[float], float, bool, bool, Dict]: observation, reward, terminated, truncated, info.
        """
        unnormalized_action = self.dictify_and_unnormalize_action(action)
        new_state, reward_inputs = self._step(state, unnormalized_action, params)
        reward, reward_terms = self.reward_model.reward(
            reward_inputs, unnormalized_action
        )
        next_time = t + self.dt

        truncated = next_time >= self.time_limit
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

    def state_to_obs(self, state: PyTree[float]) -> PyTree[float]:
        """Convert the state to an observation. This problem provides full observability, but
        the continuous observations are normalized to [-1, 1] for the sake of the agent.

        Args:
            state (PyTree[float]): system state.

        Returns:
            PyTree[float]: observations normalized to [-1, 1].
        """
        obs = np.zeros(len(self.CONT_STATES))

        def remap(key):
            value = state[key]
            return remap_range(value, self.cont_state_ranges[key], (-1.0, 1.0))

        obs = jnp.array([remap(key) for key in self.CONT_STATES])

        out = {
            "continuous": obs,
            "Hmode": state["Hmode"],
        }
        return out

    def flatten_obs(self, obs: PyTree[float]) -> jnp.ndarray:
        assert obs["Hmode"].dtype == jnp.int32
        obs_cts = obs["continuous"]
        obs_hmode = jnp.where(obs["Hmode"] == 1, 1.0, -1.0)
        
        assert obs_cts.shape == (len(self.CONT_STATES),)
        assert obs_hmode.shape == tuple()
        obs = jnp.concatenate([obs_cts, obs_hmode[None]], axis=0)
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

    @classmethod
    def create_env(cls):
        config_filepath = os.path.join(pop_down_gym.ROOT_DIR, "configs/gym.yaml")
        config = yaml.safe_load(open(config_filepath, "r"))
        model, _ = Model.create_default()
        return cls(config, model)

    @staticmethod
    def constr_labels() -> list[str]:
        constr_labels = [
            "Ip_MA",
            "Bv_dot_mag",
            "Wdot_mag",
            "beta_n",
            "beta_p",
            "li",
            "ng_frac",
            "shafranov_coeff",
            "iota95",
        ]
        return constr_labels

    @staticmethod
    def action_labels() -> list[str]:
        action_labels = [
            ""
        ]
