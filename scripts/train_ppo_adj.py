import ipdb
import jax.random as jr
import typer
from loguru import logger

from jaxrl.helpers import get_default_rew_bounds
from jaxrl.ppo import CollectorCfg, PPOCfg, PPOTrainCfg
from jaxrl.ppo_trainer import train_ppo
from jaxrl.utils.logging import set_logger_format
from jaxrl.utils.schedule import LinDecay
from pop_down_gym.pd_gym_jaxrl import PDEnv, PDEnvAdj


def main(wandb_name: str = None):
    set_logger_format()

    key = jr.PRNGKey(54123)
    rew_centers, shift_ranges, rew_min, rew_max = get_default_rew_bounds()

    offset = {"Wdot_mag": -0.5 * shift_ranges["Wdot_mag"]}

    logger.info("Constructing Env...")
    env_train = PDEnvAdj(shift_ranges=shift_ranges, limits=rew_centers)
    env_test = PDEnvAdj(shift_ranges=shift_ranges, limits=rew_centers, shift_mult=0)
    logger.info("Constructing Env... Done!")
    train_cfg = PPOTrainCfg(
        gae_lambda=0.95,
        batch_size=16_384,
        n_update_epochs=5,
        clip_ratio=0.1,
        kl_desired=0.02,
        pol_lr_max=1e-2,
        pol_lr_min=1e-5,
        normalize_V=False,
    )
    ppo_cfg = PPOCfg(
        pol_lr=3e-4,
        val_lr=3e-4,
        entropy_cf=LinDecay(1e-2, 50.0, warmup_steps=200, trans_steps=3_000),
        disc_gamma=0.99,
        pol_hid_sizes=[256, 256, 256],
        val_hid_sizes=[256, 256, 256],
        act="tanh",
        pol_type="TanhNormal",
        train_cfg=train_cfg,
        rew_scale=5e2,
        clip_grad=1.0,
    )
    collect_cfg = CollectorCfg(n_envs=2048, rollout_T=80, n_env_eval=128, rollout_T_eval=120)
    n_iters = 50_000
    train_ppo(wandb_name, key, env_train, env_test, ppo_cfg, collect_cfg, project_name="pdg_ppo_adj", n_iters=n_iters)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
