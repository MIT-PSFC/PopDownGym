import copy
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

import pop_down_gym.physics as physics
from contrax.simulate import SimFFControl
from pop_down_gym.model import Model
from pop_down_gym.reward import RewardModel
from pop_down_gym.utils import remap_range


class PopDownGym(gym.Env):
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
        self.action_space = gym.spaces.Box(
            low=-1.0 * np.ones(len(self.ACTION_RANGES)),
            high=np.ones(len(self.ACTION_RANGES)),
        )

        # Declare the space of random parameters.
        self.random_param_space = gym.spaces.Dict(
            {
                param_name: gym.spaces.Box(*param_range)
                for param_name, param_range in self.RANDOM_PARAM_RANGES.items()
            }
        )

        self.observation_space = gym.spaces.Box(
            low=-1.0 * np.ones(len(self.CONT_STATE_RANGES)),
            high=np.ones(len(self.CONT_STATE_RANGES)),
        )

    def random_initial_state(self):
        """
        Generate a random initial state.
        """
        state = copy.deepcopy(self.nominal_initial_state)
        for var, percent_variation in self.RANDOM_INITIAL_STATE_PERCENT_VAR.items():
            fractional_variation = (
                0.01 * percent_variation * self.np_random.uniform(low=-1.0, high=1.0)
            )
            state[var] = state[var] + fractional_variation * state[var]
        return state

    def reset(self, seed=None, options=None):
        # Use the following to seed self.np_random
        super().reset(seed=seed)

        # Reset time.
        self.time = 0.0

        """
        Compute a random initial state.
        """
        self.state = self.random_initial_state()

        """
        Compute a random parameter set.
        """
        self.random_params = self.random_param_space.sample()
        self.random_params = jax.tree_map(lambda val: val.squeeze(), self.random_params)

        info = {}

        return self.state_to_obs(self.state), info

    def unnormalize_action(self, action) -> dict:
        """Given an action sampled from the normalized action space, unnormalize it.

        Args:
            action (dict): _description_

        Returns:
            dict: _description_
        """
        for i, (action_name, action_val) in enumerate(action.items()):
            action_space_range = (
                self.action_space.low[i],
                self.action_space.high[i],
            )
            action[action_name] = remap_range(
                action_val, action_space_range, self.ACTION_RANGES[action_name]
            )
        return action

    @partial(jax.jit, static_argnums=(0,), backend="cpu")
    def _step(self, state, action):
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
            "Hfactor": self.random_params["Hfactor"],
            "Zeff": self.random_params["Zeff"],
            "ion_dilution": self.random_params["ion_dilution"],
            "Te_over_Ti": self.random_params["Te_over_Ti"],
            "f_dt": 0.5,
            "tau_n_factor": self.random_params["tau_n_factor"],
            "prad_mult": self.random_params["prad_mult"],
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
        PHL = self.random_params["HL_FUDGE"] * PLH
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

    def step(self, action):
        """
        Step the environment forward in time.
        """
        # Map the array of actions to a dict.
        action = {
            action_name: action[i]
            for i, action_name in enumerate(self.ACTION_RANGES.keys())
        }
        unnormalized_action = self.unnormalize_action(action)
        prev_state = self.state
        new_state, reward_inputs = self._step(prev_state, unnormalized_action)
        self.state = new_state
        reward, reward_terms = self.reward_model.reward(
            reward_inputs, unnormalized_action
        )
        self.time += self.DT

        truncated = self.time >= self.TIME_LIMIT
        hit_goal = reward_terms["hit_goal_reward"] > 0.0
        obs = self.state_to_obs(self.state)
        out_of_bounds = not self.observation_space.contains(list(obs))
        terminated = truncated or out_of_bounds or hit_goal

        info = {
            "time": self.time,
            "state": self.state,
            "action": unnormalized_action,
            "reward_inputs": reward_inputs,
            "reward_terms": reward_terms,
            "random_params": self.random_params,
            "hit_goal": hit_goal,
            "out_of_bounds": out_of_bounds,
        }

        return obs, reward, terminated, truncated, info

    def state_to_obs(self, state):
        continuous = {k: v for k, v in state.items() if k != "Hmode"}
        obs = np.zeros(len(continuous))
        # In this problem, we define the observations as the continuous states normalized to [-1, 1].
        # TODO(allenw): fair question to be asked
        for i, (key, value) in enumerate(continuous.items()):
            obs[i] = remap_range(value, self.CONT_STATE_RANGES[key], (-1.0, 1.0))
        return obs


if __name__ == "__main__":
    env = PopDownGym.create_default()
    obs, info = env.reset()
    out = env.step(env.action_space.sample())
