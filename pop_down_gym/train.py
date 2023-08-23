from pop_down_gym.pd_gym import PopDownGym
from pop_down_gym.model import Model
from pop_down_gym.visualize import VisualizeEval
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import torch as th
import yaml


def get_env_builder(cfg, seed):
    # Build the function that will build the environment.
    def build_env():
        model, _ = Model.create_default()
        env = PopDownGym(cfg, model)
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


def train(config=None, debug_mode=False):
    if config:
        project = config["user"]["project"]
        entity = config["user"]["entity"]
        if debug_mode:
            mode="disabled"
        else:
            mode="online"
        run = wandb.init(
            project=project, entity=entity, sync_tensorboard=True, config=config, mode=mode
        )
    else:
        run = wandb.init(sync_tensorboard=True)

    config = run.config
    GYM_CONFIG = os.path.join(os.path.dirname(__file__), "configs/gym.yaml")
    config["gym"] = yaml.safe_load(open(GYM_CONFIG, "r"))

    # Fill the config struct with extra data we want to record.
    n_train_envs = max(1, os.cpu_count() - config["free_cpus"])
    config["n_train_envs"] = n_train_envs

    # Directory to dump run data.
    out_dir = os.path.join(config["user"]["out_dir"], run.id)

    # Build the vectorized environment.
    if debug_mode:
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
        "MlpPolicy",
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


def debug():
    # from jax import config
    # config.update("jax_debug_nans", True)
    # config.update('jax_disable_jit', True)
    config = {
        "batch_size": 2048,
        "ent_coef": 0.0024434119085899454,
        "eval_freq": 1000,
        "free_cpus": 2,
        "gamma": 1,
        "n_eval_episodes": 10,
        "n_layers": 2,
        "n_steps_over_batch": 1,
        "total_timesteps": 5000000,
        "units_per_layer": 256,
        "debug_mode": True,
    }
    USER_FILE = os.path.join(os.path.dirname(__file__), "configs/user.yaml")
    config["user"] = yaml.safe_load(open(USER_FILE, "r"))
    train(config, debug_mode=True)


if __name__ == "__main__":
    debug()
