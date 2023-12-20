from typing import NamedTuple, TypedDict

import jax.numpy as jnp
import jax.random as jr
from loguru import logger

from jaxrl.env import Env, StepOutput
from jaxrl.env_types import Obs
from jaxrl.utils.jax_types import FloatScalar, PRNGKey
from pop_down_gym.pd_gym_stateless import PopDownGymStateless
from pop_down_gym.reward import RewardModel


class PDParamsDict(TypedDict):
    ion_dilution: float
    HL_FUDGE: float
    Hfactor: float
    Zeff: float
    Te_over_Ti: float
    tau_n_factor: float
    prad_mult: float


class PDState(NamedTuple):
    # In seconds.
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
        self.pd = PopDownGymStateless.create_env()

    def step_env(self, key: PRNGKey, state: PDState, action) -> StepOutput:
        clipped_action = action.clip(-1, 1)
        obs_tree, reward, terminated, truncated, info = self.pd.step(
            state.time, state.params, state.state, clipped_action
        )
        obs = self.pd.flatten_obs(obs_tree)
        terminated = info["out_of_bounds"] | info["hit_goal"]
        new_state = PDState(info["time"], state.params, info["state"])

        params_vec = self._params_to_obsvec(state.params)
        obs_priv = jnp.concatenate([obs, params_vec], axis=0)
        assert obs_priv.ndim == 1

        info_ = {k: info[k] for k in ["time", "reward_inputs", "reward_terms", "hit_goal", "out_of_bounds"]}

        return StepOutput(obs, obs_priv, new_state, reward, terminated, truncated, info_)

    def _params_to_obsvec(self, params: PDParamsDict) -> Obs:
        obs = []
        for k, obs_range in self.pd.random_param_ranges.items():
            # Normalize to [0, 1]
            obs_normalized = (params[k] - obs_range[0]) / (obs_range[1] - obs_range[0])
            # Normalize to [-1, 1]
            obs_normalized = 2 * obs_normalized - 1
            obs.append(obs_normalized)

        return jnp.array(obs)

    def reset_env(self, key: PRNGKey) -> tuple[Obs, Obs, PDState]:
        params, state, obs_tree, info = self.pd.reset(key)
        env_state = PDState(
            info["time"],
            params,
            state,
        )
        obs = self.pd.flatten_obs(obs_tree)
        params_vec = self._params_to_obsvec(env_state.params)
        obs_priv = jnp.concatenate([obs, params_vec], axis=0)
        assert obs_priv.ndim == 1
        return obs, obs_priv, env_state

    @property
    def n_actions(self) -> int:
        return self.pd.n_actions

    @property
    def constr_labels(self):
        return PopDownGymStateless.constr_labels()


class PDEnvAdj(Env):
    def __init__(
        self,
        shift_ranges: dict = None,
        offset: dict = None,
        shift_mult: float = 1.0,
        limits: dict = None,
        env_params: dict = None,
    ):
        if shift_ranges is None:
            shift_ranges = {"Bv_dot_mag": 0.1, "beta_p": 0.1, "li": 1.0}
        if offset is None:
            offset = {}
        if env_params is None:
            env_params = {}
        self.pd = PopDownGymStateless.create_env()
        if limits is not None:
            logger.info("Overriding reward model limits:")
            for k, new_limit in limits.items():
                old_limit = self.pd.reward_model.limits[k]
                logger.info("{:12} {} -> {}".format(k, old_limit, new_limit))
                assert k in self.pd.reward_model.limits
                self.pd.reward_model.limits[k] = new_limit

        self.shift_ranges = shift_ranges
        self.offset = offset
        self.shift_mult = shift_mult

        self.env_params = env_params

    @property
    def dt(self):
        return self.pd.dt

    def get_obs(self, obs_tree: dict[str, jnp.array], info: dict, shift_obs):
        return jnp.concatenate([self.pd.flatten_obs(obs_tree), shift_obs])

    def _get_obs_priv(self, obs_tree, info, shift_obs, params_vec):
        obs = self.pd.flatten_obs(obs_tree)
        obs_priv = jnp.concatenate([obs, shift_obs, params_vec], axis=0)
        return obs_priv

    def get_reward_model(self, shifts: dict[str, FloatScalar] = None) -> RewardModel:
        if shifts is None:
            shifts = self.offset

        params = self.pd.reward_model.params.copy()
        params["limits"] = params["limits"].copy()
        for k, v in shifts.items():
            assert k in params["limits"]
            params["limits"][k] += v
        return RewardModel(params)

    def step_env(self, key: PRNGKey, state: PDAdjState, action) -> StepOutput:
        obs_tree, reward, terminated, truncated, info = self.pd.step(state.time, state.params, state.state, action)
        terminated = info["out_of_bounds"] | info["hit_goal"]
        new_state = PDAdjState(info["time"], state.params, info["state"], state.shifts)

        # Recompute the reward.
        reward_model = self.get_reward_model(state.shifts)

        unnormalized_action = self.pd.dictify_and_unnormalize_action(action)
        reward, reward_terms = reward_model.reward(info["reward_inputs"], unnormalized_action)

        info_ = {k: info[k] for k in ["time", "reward_inputs", "reward_terms", "hit_goal", "out_of_bounds"]}
        info_["orig_reward_terms"] = info_["reward_terms"]
        info_["reward_terms"] = reward_terms

        # Make the shift ranges observable. [-1, 1].
        shift_obs = self.get_shift_obs(state.shifts)
        obs = self.get_obs(obs_tree, info, shift_obs)

        params_vec = self._params_to_obsvec(state.params)
        obs_priv = self._get_obs_priv(obs_tree, info, shift_obs, params_vec)

        return StepOutput(obs, obs_priv, new_state, reward, terminated, truncated, info_)

    def get_shift_obs(self, state_shifts):
        return jnp.array([state_shifts[k] / shift for k, shift in self.shift_ranges.items()])

    def _params_to_obsvec(self, params: PDParamsDict) -> Obs:
        obs = []
        for k, obs_range in self.pd.random_param_ranges.items():
            # Normalize to [0, 1]
            obs_normalized = (params[k] - obs_range[0]) / (obs_range[1] - obs_range[0])
            # Normalize to [-1, 1]
            obs_normalized = 2 * obs_normalized - 1
            obs.append(obs_normalized)

        return jnp.array(obs)

    def reset_env(self, key: PRNGKey) -> tuple[Obs, Obs, PDAdjState]:
        key_pd, key_shifts = jr.split(key, 2)
        params, state, obs_tree, info = self.pd.reset(key_pd)

        shifts_arr = self.shift_mult * jr.uniform(key_shifts, (len(self.shift_ranges),), minval=-1, maxval=1)

        shifts = {k: v * self.shift_ranges[k] for k, v in zip(self.shift_ranges.keys(), shifts_arr)}
        for k, v in self.offset.items():
            shifts[k] = shifts[k] + v

        for k, v in self.env_params.items():
            assert k in params
            logger.info("Overriding env param {:12}: {} -> {}".format(k, params[k], v))
            params[k] = v

        env_state = PDAdjState(info["time"], params, state, shifts)

        # Make the shift ranges observable. [-1, 1].
        shift_obs = self.get_shift_obs(env_state.shifts)
        obs = self.get_obs(obs_tree, info, shift_obs)

        params_vec = self._params_to_obsvec(env_state.params)
        obs_priv = self._get_obs_priv(obs_tree, info, shift_obs, params_vec)
        assert obs_priv.ndim == 1
        return obs, obs_priv, env_state

    @property
    def n_actions(self) -> int:
        return self.pd.n_actions

    @property
    def constr_labels(self):
        return PopDownGymStateless.constr_labels()


class PDEnvFFAdj(PDEnvAdj):
    """PDEnv, but only the time and shift ranges are observable."""

    def get_obs(self, obs_tree: dict[str, jnp.array], info: dict, shift_obs):
        time_s = info["time"]
        assert isinstance(time_s, float) or time_s.shape == tuple()
        time_s = jnp.array([time_s])
        return jnp.concatenate([time_s, shift_obs], axis=0)

    def _get_obs_priv(self, obs_tree, info, shift_obs, params_vec):
        time_s = info["time"]
        time_s = jnp.array([time_s])
        assert time_s.shape == (1,)

        obs = self.pd.flatten_obs(obs_tree)
        obs_priv = jnp.concatenate([time_s, obs, shift_obs, params_vec], axis=0)
        logger.info(
            "time: {} obs: {}, shift_obs: {}, params_vec: {} -> obs_priv: {}".format(
                time_s.shape, obs.shape, shift_obs.shape, params_vec.shape, obs_priv.shape
            )
        )

        return obs_priv
