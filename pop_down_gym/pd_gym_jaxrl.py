from typing import NamedTuple, TypedDict

import jax.numpy as jnp
import jax.random as jr

from jaxrl.env import Env, StepOutput
from jaxrl.env_types import Obs
from jaxrl.utils.jax_types import FloatScalar, PRNGKey
from pop_down_gym.pd_gym_stateless import PopDownGymStateless, make_pd_stateless
from pop_down_gym.reward import RewardModel
from loguru import logger


class PDParamsDict(TypedDict):
    ion_dilution: float
    HL_FUDGE: float
    Hfactor: float
    Zeff: float
    Te_over_Ti: float
    tau_n_factor: float
    prad_mult: float


class PDState(NamedTuple):
    time: FloatScalar
    params: PDParamsDict
    state: dict


class PDAdjState(NamedTuple):
    time: FloatScalar
    params: dict
    state: dict
    shifts: dict


class PDEnv(Env):
    def __init__(self):
        self.pd, _ = make_pd_stateless()

    def step_env(self, key: PRNGKey, state: PDState, action) -> StepOutput:
        clipped_action = action.clip(-1, 1)
        obs, reward, terminated, truncated, info = self.pd.step(state.time, state.params, state.state, clipped_action)
        terminated = info["out_of_bounds"] | info["hit_goal"]
        new_state = PDState(info["time"], state.params, info["state"])

        params_vec = self._params_to_obsvec(state.params)
        obs_priv = jnp.concatenate([obs, params_vec], axis=0)
        assert obs_priv.ndim == 1

        info_ = {k: info[k] for k in ["time", "reward_inputs", "reward_terms", "hit_goal", "out_of_bounds"]}

        return StepOutput(obs, obs_priv, new_state, reward, terminated, truncated, info_)

    def _params_to_obsvec(self, params: PDParamsDict) -> Obs:
        obs = []
        for k, obs_range in PopDownGymStateless.RANDOM_PARAM_RANGES.items():
            # Normalize to [0, 1]
            obs_normalized = (params[k] - obs_range[0]) / (obs_range[1] - obs_range[0])
            # Normalize to [-1, 1]
            obs_normalized = 2 * obs_normalized - 1
            obs.append(obs_normalized)

        return jnp.array(obs)

    def reset_env(self, key: PRNGKey) -> tuple[Obs, Obs, PDState]:
        params, state, obs, info = self.pd.reset(key)
        env_state = PDState(
            info["time"],
            params,
            state,
        )
        params_vec = self._params_to_obsvec(env_state.params)
        obs_priv = jnp.concatenate([obs, params_vec], axis=0)
        assert obs_priv.ndim == 1
        return obs, obs_priv, env_state

    @property
    def n_actions(self) -> int:
        return self.pd.n_actions


class PDEnvAdj(Env):
    def __init__(self, shift_ranges: dict = None, offset: dict = None, shift_mult: float = 1.0, limits: dict = None):
        if shift_ranges is None:
            shift_ranges = {"Bv_dot_mag": 0.1, "beta_p": 0.1, "li": 1.0}
        if offset is None:
            offset = {}
        self.pd, _ = make_pd_stateless()
        if limits is not None:
            logger.info("Overriding reward model limits:")
            for k, new_limit in limits.items():
                old_limit = self.pd.reward_model.limits[k]
                logger.info("{:12} {} -> {}".format(k, old_limit, new_limit))
                self.pd.reward_model.limits[k] = new_limit

        self.shift_ranges = shift_ranges
        self.offset = offset
        self.shift_mult = shift_mult

    def step_env(self, key: PRNGKey, state: PDAdjState, action) -> StepOutput:
        obs, reward, terminated, truncated, info = self.pd.step(state.time, state.params, state.state, action)
        terminated = info["out_of_bounds"] | info["hit_goal"]
        new_state = PDAdjState(info["time"], state.params, info["state"], state.shifts)

        # Recompute the reward.
        action = {action_name: action[i] for i, action_name in enumerate(self.pd.ACTION_RANGES.keys())}
        unnormalized_action = self.pd.unnormalize_action(action)

        params = self.pd.reward_model.params.copy()
        params["limits"] = params["limits"].copy()
        for k, v in state.shifts.items():
            params["limits"][k] += v
        reward_model = RewardModel(params)
        reward, reward_terms = reward_model.reward(info["reward_inputs"], unnormalized_action)

        info_ = {k: info[k] for k in ["time", "reward_inputs", "reward_terms", "hit_goal", "out_of_bounds"]}
        info_["orig_reward_terms"] = info_["reward_terms"]
        info_["reward_terms"] = reward_terms

        # Make the shift ranges observable. [-1, 1].
        obs = self.add_to_obs(obs, state.shifts)

        params_vec = self._params_to_obsvec(state.params)
        obs_priv = jnp.concatenate([obs, params_vec], axis=0)

        return StepOutput(obs, obs_priv, new_state, reward, terminated, truncated, info_)

    def add_to_obs(self, obs, shifts):
        shifts = jnp.array([shifts[k] / shift for k, shift in self.shift_ranges.items()])
        obs = jnp.concatenate([obs, shifts])
        return obs

    def _params_to_obsvec(self, params: PDParamsDict) -> Obs:
        obs = []
        for k, obs_range in PopDownGymStateless.RANDOM_PARAM_RANGES.items():
            # Normalize to [0, 1]
            obs_normalized = (params[k] - obs_range[0]) / (obs_range[1] - obs_range[0])
            # Normalize to [-1, 1]
            obs_normalized = 2 * obs_normalized - 1
            obs.append(obs_normalized)

        return jnp.array(obs)

    def reset_env(self, key: PRNGKey) -> tuple[Obs, Obs, PDAdjState]:
        key_pd, key_shifts = jr.split(key, 2)
        params, state, obs, info = self.pd.reset(key_pd)

        shifts_arr = self.shift_mult * jr.uniform(key_shifts, (len(self.shift_ranges),), minval=-1, maxval=1)

        shifts = {k: v * self.shift_ranges[k] for k, v in zip(self.shift_ranges.keys(), shifts_arr)}
        for k, v in self.offset.items():
            shifts[k] = shifts[k] + v
        env_state = PDAdjState(info["time"], params, state, shifts)

        # Make the shift ranges observable. [-1, 1].
        obs = self.add_to_obs(obs, shifts)

        params_vec = self._params_to_obsvec(env_state.params)
        obs_priv = jnp.concatenate([obs, params_vec], axis=0)
        assert obs_priv.ndim == 1
        return obs, obs_priv, env_state

    @property
    def n_actions(self) -> int:
        return self.pd.n_actions
