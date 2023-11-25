from typing import Any, NamedTuple, TypeVar

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from jaxrl.utils.jax_types import AnyFloat, BoolScalar, PRNGKey

_Obs = TypeVar("_Obs")
_State = TypeVar("_State")


class StepOutput(NamedTuple):
    obs: _Obs
    # Observations, but with privileged information.
    obs_priv: _Obs
    state: _State
    reward: AnyFloat
    terminated: BoolScalar
    truncated: BoolScalar
    info: Any


class Env:
    def step_autoreset(self, key: PRNGKey, state, action) -> StepOutput:
        """Automatically reset if either truncated or terminated. If so, returns the new state and obs."""
        key, key_reset = jr.split(key, 2)
        obs_st, obsp_st, state_st, reward, terminated, truncated, info = self.step_env(key, state, action)
        obs_re, obsp_re, state_re = self.reset_env(key_reset)

        should_reset = terminated | truncated
        stateobs_st = (state_st, (obs_st, obsp_st))
        stateobs_re = (state_re, (obs_re, obsp_re))
        state, obs_tup = jtu.tree_map(lambda x, y: jnp.where(should_reset, x, y), stateobs_re, stateobs_st)
        obs, obs_priv = obs_tup

        return StepOutput(obs, obs_priv, state, reward, terminated, truncated, info)

    def step_env(self, key: PRNGKey, state, action) -> StepOutput:
        raise NotImplementedError("")

    def reset(self, key: PRNGKey) -> tuple[_Obs, _Obs, _State]:
        return self.reset_env(key)

    def reset_env(self, key: PRNGKey) -> tuple[_Obs, _Obs, _State]:
        raise NotImplementedError("")

    @property
    def n_actions(self) -> int:
        raise NotImplementedError("")
