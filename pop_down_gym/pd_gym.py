import os

import gymnasium as gym
import jax
import numpy as np
import yaml

from pop_down_gym.model import Model
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


class PopDownGym(gym.Env):
    """A Gym-compatible wrapper around the stateless gym environment."""
    

    def __init__(self, cfg: dict, model: Model):
        self.stateless_env = PopDownGymStateless(cfg, model)

        # Declare the normalized action space.
        self.action_space = gym.spaces.Box(
            low=-1.0 * np.ones(len(self.stateless_env.ACTION_RANGES)),
            high=np.ones(len(self.stateless_env.ACTION_RANGES)),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "continuous": gym.spaces.Box(
                    low=-1.0 * np.ones(len(self.stateless_env.CONT_STATE_RANGES)),
                    high=np.ones(len(self.stateless_env.CONT_STATE_RANGES)),
                ),
                "Hmode": gym.spaces.Discrete(2),
            }
        )

    def random_initial_state(self):
        """
        Generate a random initial state.
        """
        # Get a JAX key from a random seed.
        key = jax.random.PRNGKey(np.random.randint(0, 2 ** 32 - 1))
        return self.stateless_env.sample_state(key)

    def reset(self, seed=None, options=None):
        # Use the following to seed self.np_random
        super().reset(seed=seed)

        # Get a JAX key from a random seed.
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        key = jax.random.PRNGKey(seed)

        # Reset time.
        self.time = 0.0

        # Sample random initial state and parameters
        params, state, obs, info = self.stateless_env.reset(key)
        self.state = state
        self.random_params = params

        info = {}

        return obs, info

    def step(self, action):
        """
        Step the environment forward in time.
        """
        obs, reward, terminated, truncated, info = self.stateless_env.step(
            self.time, self.random_params, self.state, action
        )
        self.time = info["time"]

        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    config_filepath = os.path.join(os.path.dirname(__file__), "configs/gym.yaml")
    config = yaml.safe_load(open(config_filepath, "r"))
    model, _ = Model.create_default()
    env = PopDownGym(config, model)
    obs, info = env.reset()
    out = env.step(env.action_space.sample())
