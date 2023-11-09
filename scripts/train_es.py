"""Optimize a control policy using evolutionary strategy."""
import os

import equinox as eqx
import evosax as es
import jax
import yaml
from tqdm import tqdm

from pop_down_gym.model import Model
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


class MLP(eqx.Module):
    layers: list

    def __init__(self, key, input_dims, hidden_dims, output_dims):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(input_dims, hidden_dims, key=key1),
            eqx.nn.Linear(hidden_dims, hidden_dims, key=key2),
            eqx.nn.Linear(hidden_dims, output_dims, key=key3),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jax.nn.tanh(self.layers[-1](x))


def create_env():
    config_filepath = os.path.join(
        os.path.dirname(__file__), "../pop_down_gym/configs/gym.yaml"
    )
    config = yaml.safe_load(open(config_filepath, "r"))
    model, _ = Model.create_default()
    env = PopDownGymStateless(config, model)
    return env


def rollout_closed_loop(prng_key, env, policy, steps=100):
    """Simulate the policy in the environment."""
    # Sample random parameters and initial state
    params, initial_state, initial_obs, _ = env.reset(prng_key)

    # Define a step function to simulate using scan
    def scan_step(carry, _):
        # Unpack the carry
        state, obs, t = carry

        # Evaluate the policy
        action = policy(obs)

        # Step the environment
        obs, reward, terminated, _, info = env.step(t, params, state, action)

        # If we've terminated, don't update the state
        next_state = jax.lax.cond(
            terminated, lambda _: state, lambda _: info["state"], None
        )
        next_time = jax.lax.cond(terminated, lambda _: t, lambda _: info["time"], None)

        # prepare the carry for the next iteration
        carry = (next_state, obs, next_time)
        output = (reward, next_state, next_time, info["reward_inputs"])

        return carry, output

    # Simulate the trajectory
    _, (rewards, states, t, reward_inputs) = jax.lax.scan(
        scan_step, (initial_state, initial_obs, 0.0), None, length=steps
    )

    return rewards.mean(), states, t, reward_inputs, rewards


if __name__ == "__main__":
    prng_key = jax.random.PRNGKey(0)

    # Load the environment
    env = create_env()

    # Hyperparams
    hidden_dims = 256
    simulation_steps = 100
    num_generations = 200
    popsize = int(4e1)
    num_eval_rollouts = int(1e3)
    lrate_init = 1e-2

    # Define the fitness function
    fitness_single_rollout = lambda key, policy: rollout_closed_loop(
        key, env, policy, steps=simulation_steps
    )[0]

    def fitness_multiple_rollouts(key, policy):
        keys = jax.random.split(key, num_eval_rollouts)
        return jax.vmap(fitness_single_rollout, in_axes=(0, None))(keys, policy).mean()

    def population_fitness(key, population):
        # The population has a leading axis for devices and a second axis
        # for vectorized dimension
        return jax.pmap(
            jax.jit(jax.vmap(fitness_multiple_rollouts, in_axes=(None, 0))),
            in_axes=(None, 0),
        )(key, population)

    # Create an MLP
    prng_key, policy_key = jax.random.split(prng_key)
    mlp = MLP(policy_key, env.n_obs, hidden_dims, env.n_actions)

    # Set up ES
    param_reshaper = es.ParameterReshaper(mlp)
    strategy = es.OpenES(
        popsize=popsize,
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        lrate_init=lrate_init,
    )
    es_logging = es.ESLog(
        param_reshaper.total_params,
        num_generations=num_generations,
        top_k=5,
        maximize=True,
    )
    log = es_logging.initialize()
    fit_shaper = es.FitnessShaper(centered_rank=True, w_decay=0.0, maximize=True)
    prng_key, es_key = jax.random.split(prng_key)
    state = strategy.initialize(es_key)

    # Run the evolution
    pbar = tqdm(range(num_generations))
    for gen in pbar:
        prng_key, prng_init, prng_ask, prng_eval = jax.random.split(prng_key, 4)

        # Generate a new population
        x, state = strategy.ask(prng_ask, state)

        # Evaluate the population
        reshaped_params = param_reshaper.reshape(x)
        fitness = population_fitness(prng_eval, reshaped_params).reshape(-1)
        fit_re = fit_shaper.apply(x, fitness)

        # Update the population
        state = strategy.tell(x, fit_re, state)

        # Log
        log = es_logging.update(log, x, fitness)
        pbar.set_description(f"Performance: {log['log_top_1'][gen]:.3f}")

    # Plot and save to file
    fig, ax = es_logging.plot(log, "PopDownGym")
    fig.savefig("tmp/es.png")

    # Save the best policy and top-5 policies
    one_device_reshaper = es.ParameterReshaper(mlp, n_devices=1)
    top5_policies = one_device_reshaper.reshape(log["top_params"])
    best_policy = jax.tree_map(lambda x: x[0], top5_policies)
    eqx.tree_serialise_leaves("tmp/es_best_policy.eqx", best_policy)
    eqx.tree_serialise_leaves("tmp/es_best_5_policies.eqx", top5_policies)
