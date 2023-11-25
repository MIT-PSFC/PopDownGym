import jax

from pop_down_gym.benchmark import benchmark_simulate
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def test_env_sample_state():
    # Create an environment
    env = PopDownGymStateless.create_env()

    # Sample a random state
    key = jax.random.PRNGKey(0)
    state = env.sample_state(key)

    assert state is not None
    assert isinstance(state, dict)


def test_env_sample_params():
    # Create an environment
    env = PopDownGymStateless.create_env()

    # Sample a random state
    key = jax.random.PRNGKey(0)
    params = env.sample_params(key)

    assert params is not None
    assert isinstance(params, dict)

def test_env_n_obs():
    env = PopDownGymStateless.create_env()
    assert env.n_obs == len(env.observation_space["continuous"]) + 1

def test_benchmark_simulate():
    batch_sizes = [1, 16]
    results = benchmark_simulate(batch_sizes, num_trials=1)
    assert results is not None