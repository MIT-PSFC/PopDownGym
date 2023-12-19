
import click
import jax
import jax.numpy as jnp
import pandas as pd
import equinox as eqx
from stable_baselines3 import PPO
import os
import yaml
import pop_down_gym
from pop_down_gym.train import get_env_builder
from pop_down_gym.raptor.raptor_rd_gym import RaptorRDGym
from pop_down_gym.raptor.utils import convert_to_df
from pop_down_gym.pd_gym import PopDownGym
from pop_down_gym.scripts.train_es import MLP as ES_MLP
from pop_down_gym.scripts.example_load_ppo_adj_ckpt import default_build_ppo
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

class PolicyInterface:
    def __init__(self):
        self.model, self.train_env = PolicyInterface.load_model_and_train_env()

        self.train_env = vec_env.envs[0]
        self.constraint_limits = self.train_env.stateless_env.reward_model.limits
        if model_type == "PPO_SB3":
            model = PPO.load(model_path, env=vec_env)
            self.model_fn = lambda obs: model.predict(obs)[0]
        elif model_type == "ES_DAWSON":
            prng_key = jax.random.PRNGKey(0)
            # Load the best policy trained using ES
            hidden_dims = 512
            hidden_layers = 4
            prng_key, policy_key = jax.random.split(prng_key)
            mlp = ES_MLP(policy_key, self.train_env.stateless_env.n_obs, hidden_layers, hidden_dims, self.train_env.stateless_env.n_actions)
            mlp = eqx.tree_deserialise_leaves(model_path, mlp)                
            self.model_fn = mlp
        elif model_type == "PPO_OSO":
            ppo, env, offset_dict, tmp_dir = default_build_ppo()
            def model_fn(obs):
                obs = self.train_env.stateless_env.flatten_obs(obs)
                constraint_shifts = jnp.zeros(8)
                return ppo.act(jnp.concatenate([obs, constraint_shifts]))
            self.model_fn = model_fn
        else:
            raise ValueError("model_type must be one of PPO_SB3, ES_DAWSON, PPO_OSO")
        
    @staticmethod
    def get_env_builder(seed):
        def build_env():
            env = PopDownGym.create_env()
            env.reset(seed=seed)
            return env
        return build_env
    
    def state_to_action(self, train_env_state):
        obs = self.train_env.state_to_obs(train_env_state)
        action = self.model.predict(obs)[0]
        unnormalized_action = self.train_env.dictify_and_unnormalize_action(action)
        return action, unnormalized_action
    
    def step_train_env(self, action):
        return self.train_env.step(action)
    

def run_with_raptor_loop(raptor_gym, policy_interface, max_steps = 140):
    actions = []
    for i in range(max_steps):
        obs_for_pd_gym = raptor_gym.obs_for_pd_gym()
        # We have reached the goal.
        if obs_for_pd_gym["Ip_MA"] < 2.0:
            break
        action, unnormalized_action = policy_interface.state_to_action(obs_for_pd_gym)
        
        raptor_action_input = {
            "dIp_dt": 1e6 * unnormalized_action["dIp_dt"],
            "dPaux_dt": 1e6 * unnormalized_action["dPaux_dt"],
            "fueling19": unnormalized_action["fueling19"],
            "dgs_dt": unnormalized_action["dgs_dt"],
        }
        actions.append({"time": raptor_gym.time, **raptor_action_input})
        raptor_gym.step(raptor_action_input)
    
    raptor_out = raptor_gym.raptor_out()
    return raptor_out, actions

@click.command()
@click.option("--model_path", type=str, default="/home/awang/Scratch/rd_rl/20231004_best/best_model.zip")
@click.option("--model_type", type=str, default="PPO_SB3")
@click.option("--raptor_path", type=str, default="/home/awang/raptor")
def main(model_path, model_type, raptor_path):
    policy_interface = PolicyInterface(model_path=model_path, model_type=model_type)
    gym_dt = policy_interface.train_env.stateless_env.dt
    raptor_dt = 0.005

    
    raptor_steps_per_gym_step = gym_dt / raptor_dt
    # Assert near int.
    assert abs(raptor_steps_per_gym_step - round(raptor_steps_per_gym_step)) < 1e-6
    raptor_steps_per_gym_step = int(round(raptor_steps_per_gym_step))

    raptor_gym = RaptorRDGym(raptor_path, raptor_dt, raptor_steps_per_gym_step)
    raptor_gym.reset()
    assert raptor_dt * raptor_steps_per_gym_step == policy_interface.train_env.stateless_env.dt
    raptor_out, actions = run_with_raptor_loop(raptor_gym, policy_interface)
    raptor_out_df = convert_to_df(raptor_out)
    df = pd.DataFrame(actions)
    df=df.set_index('time')
    df=df.join(raptor_out_df,how='inner') 
    df.attrs['constraint_limits'] = policy_interface.constraint_limits
    df.to_pickle("sim2sim.pkl")

if __name__ == "__main__":
    main()
