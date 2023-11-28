import click
import jax
import pandas as pd
import equinox as eqx
from stable_baselines3 import PPO
from pop_down_gym.raptor.raptor_rd_gym import RaptorRDGym
from pop_down_gym.raptor.visualize import plot_df
from pop_down_gym.pd_gym import PopDownGym
from pop_down_gym.scripts.train_es import MLP as ES_MLP
from stable_baselines3.common.vec_env import DummyVecEnv

class PolicyInterface:
    def __init__(self, model_path, model_type):
        vec_env = DummyVecEnv([PolicyInterface.get_env_builder(0)])

        self.train_env = vec_env.envs[0]
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
            pass
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
        obs = self.train_env.stateless_env.state_to_obs(train_env_state)
        action = self.model_fn(obs)
        unnormalized_action = self.train_env.stateless_env.dictify_and_unnormalize_action(action)
        return action, unnormalized_action
    
def run_with_raptor_loop(raptor_gym, policy_interface, max_steps = 140):
    states = []
    for i in range(max_steps):
        state_for_pd_gym = raptor_gym.state_for_pd_gym()
        states.append(state_for_pd_gym)
        if state_for_pd_gym["Ip_MA"] < 2.0:
            break
        action, unnormalized_action = policy_interface.state_to_action(state_for_pd_gym)
        
        raptor_action_input = {
            "dIp_dt": 1e6 * unnormalized_action["dIp_dt"],
            "dPaux_dt": 1e6 * unnormalized_action["dPaux_dt"],
            "fueling19": unnormalized_action["fueling19"],
            "dgs_dt": unnormalized_action["dgs_dt"],
        }
        raptor_gym.step(raptor_action_input)
    
    raptor_out = raptor_gym.raptor_out()
    return raptor_out, states

@click.command()
@click.option("--model_path", type=str, default="/home/awang/Scratch/rd_rl/20231004_best/best_model.zip")
@click.option("--model_type", type=str, default="PPO_SB3")
@click.option("--raptor_path", type=str, default="/home/awang/raptor")
def main(model_path, model_type, raptor_path):
    policy_interface = PolicyInterface(model_path=model_path, model_type=model_type)

    raptor_dt = 0.01
    raptor_steps_per_gym_step = 5
    raptor_gym = RaptorRDGym(raptor_path, 1e-2, 5)
    raptor_gym.reset()
    assert raptor_dt * raptor_steps_per_gym_step == policy_interface.train_env.stateless_env.dt
    raptor_out, states = run_with_raptor_loop(raptor_gym, policy_interface)
    df = pd.DataFrame(states)
    df = df.applymap(lambda x: 1 if x == True else x)
    raptor_gym.save_out("test.mat")

if __name__ == "__main__":
    main()