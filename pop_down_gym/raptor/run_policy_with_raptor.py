from stable_baselines3 import PPO
import os
import yaml
import pop_down_gym
from pop_down_gym.train import get_env_builder
from pop_down_gym.raptor.raptor_rd_gym import RaptorRDGym
from pop_down_gym.raptor.visualize import plot_df
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

class PolicyInterface:
    def __init__(self):
        self.model, self.train_env = PolicyInterface.load_model_and_train_env()

    @staticmethod
    def load_model_and_train_env():
        GYM_CONFIG = os.path.join(pop_down_gym.ROOT_DIR, "configs/gym.yaml")
        gym_config = yaml.safe_load(open(GYM_CONFIG, "r"))
        env = DummyVecEnv([get_env_builder(gym_config, 0)])
        model = PPO.load("/home/awang/Scratch/rd_rl/20231004_best/best_model.zip", env=env)
        return model, env.envs[0]

    def state_to_action(self, train_env_state):
        obs = self.train_env.state_to_obs(train_env_state)
        action = self.model.predict(obs)[0]
        unnormalized_action = self.train_env.dictify_and_unnormalize_action(action)
        return action, unnormalized_action
    
    def step_train_env(self, action):
        return self.train_env.step(action)
    

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


if __name__ == "__main__":
    policy_interface = PolicyInterface()

    raptor_dt = 0.01
    raptor_steps_per_gym_step = 5
    raptor_gym = RaptorRDGym("/home/awang/raptor", 1e-2, 5)
    raptor_gym.reset()
    assert raptor_dt * raptor_steps_per_gym_step == policy_interface.train_env.dt
    raptor_out, states = run_with_raptor_loop(raptor_gym, policy_interface)
    df = pd.DataFrame(states)
    df = df.applymap(lambda x: 1 if x == True else x)
    raptor_gym.save_out("test.mat")
    plot_df(df)