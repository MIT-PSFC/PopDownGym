import math
import os

import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from pop_down_gym import ROOT_DIR
from pop_down_gym.model import Model
from pop_down_gym.pd_gym import PopDownGym
from pop_down_gym.visualize import VisualizeEval



def sweep(project_name, out_dir, total_timesteps):
    # Directory containing this script.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    program = os.path.join(dir_path, "train.py")

    sweep_configuration = {
        "name": "sweep",
        "method": "bayes",
        "metric": {"name": "eval/mean_reward", "goal": "maximize"},
        "program": program,
        "parameters": {
            "out_dir": {"value": out_dir},
            "total_timesteps": {"value": total_timesteps},
            "free_cpu_frac": {"value": 0.2},
            "eval_freq": {"value": 1e4},
            "n_eval_episodes": {"value": 10},
            "gamma": {"value": 1.0},
            "batch_size": {"values": [1024, 2048, 4096]},
            "n_steps_over_batch": {"values": [1, 2, 3]},
            "ent_coef": {"min": 0.0, "max": 0.02},
            "n_layers": {"values": [2, 3]},
            "units_per_layer": {"values": [64, 128, 256]},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project=project_name, entity="allen_adastra"
    )
    print(f"sweep_id: {sweep_id}")

def get_env_builder(cfg, seed):
    # Build the function that will build the environment.
    def build_env():
        env = PopDownGym.create_env()
        env.reset(seed=seed)
        return Monitor(env)

    return build_env


def build_policy_kwargs(n_layers: int, units_per_layer: int):
    """Build the dictionary of policy kwargs to specify
    the architecture of the policy network.

    Args:
        n_layers (int): number of layers in the policy network.
        units_per_layer (int): number of units per layer in the policy network.

    Returns:
        dict:
    """
    policy_kwargs = {
        "net_arch": dict(
            pi=[units_per_layer for _ in range(n_layers)],
            vf=[units_per_layer for _ in range(n_layers)],
        ),
        "activation_fn": th.nn.ReLU,
    }
    return policy_kwargs


def train(config={}):
    USER_FILE = os.path.join(ROOT_DIR, "configs/user.yaml")
    config["user"] = yaml.safe_load(open(USER_FILE, "r"))
    GYM_CONFIG = os.path.join(ROOT_DIR, "configs/gym.yaml")
    config["gym"] = yaml.safe_load(open(GYM_CONFIG, "r"))
    if config:
        project = config["user"]["project"]
        entity = config["user"]["entity"]
        run = wandb.init(
            project=project,
            entity=entity,
            sync_tensorboard=True,
            config=config,
            mode="disabled" if config["debug_env"] else "online",
        )
    else:
        run = wandb.init(sync_tensorboard=True)

    config = run.config

    # Fill the config struct with extra data we want to record.

    n_train_envs = max(1, math.floor(config["free_cpu_frac"] * os.cpu_count()))
    config["n_train_envs"] = n_train_envs

    # Directory to dump run data.
    out_dir = os.path.join(config["user"]["out_dir"], run.id)

    # Build the vectorized environment.
    if config["debug_env"]:
        vec_env = DummyVecEnv([get_env_builder(config["gym"], 0)])
    else:
        vec_env = SubprocVecEnv(
            [get_env_builder(config["gym"], i) for i in range(n_train_envs)]
        )
    # Initialize callbacks.
    wandb_cb = WandbCallback(model_save_freq=1e4, model_save_path=out_dir, verbose=2)
    eval_wandb_cb = VisualizeEval(vec_env, config["gym"]["reward"]["limits"], run)
    eval_cb = EvalCallback(
        vec_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        callback_on_new_best=eval_wandb_cb,
        n_eval_episodes=config["n_eval_episodes"],
        eval_freq=config["eval_freq"],  # Number of steps per environment.
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        seed=42,
        verbose=1,
        tensorboard_log=out_dir,
        device="cpu",  # Running into a torch<->cuda bug rn...
        n_steps=config["n_steps_over_batch"] * config["batch_size"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        policy_kwargs=build_policy_kwargs(
            config["n_layers"], config["units_per_layer"]
        ),
        use_sde=False,  # TODO(allenw): crashing with this on.
    )

    # Train the model.
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_cb, eval_cb],
    )
    run.finish()


def example():
    config = {
        "batch_size": 2048,
        "ent_coef": 0.0024434119085899454,  # Identified via hyperparameter search.
        "eval_freq": 5000,
        "free_cpu_frac": 0.2,
        "gamma": 1,
        "n_eval_episodes": 20,
        "n_layers": 3,
        "n_steps_over_batch": 1,
        "total_timesteps": 10000000,
        "units_per_layer": 128,
        "debug_env": False,
    }
    train(config)


if __name__ == "__main__":
    example()