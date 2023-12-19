import functools as ft
import pathlib

import ipdb
import jax.random as jr
import typer
from loguru import logger

from jaxrl.helpers import get_default_rew_bounds, load_ppo
from jaxrl.ppo import CollectorCfg, PPOCfg, PPOTrainCfg
from jaxrl.ppo_trainer import train_ppo
from jaxrl.utils.logging import set_logger_format
from jaxrl.utils.schedule import LinDecay
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj, PDEnvFFAdj
from pop_down_gym.ppo_ff import make_ppo_ff_bc


def main(ckpt_dir: pathlib.Path, wandb_name: str = None):
    set_logger_format()

    key = jr.PRNGKey(54123)
    rew_centers, shift_ranges, _, _ = get_default_rew_bounds()

    logger.info("Constructing Env...")
    env_train = PDEnvFFAdj(shift_ranges=shift_ranges, limits=rew_centers)
    env_test = PDEnvFFAdj(shift_ranges=shift_ranges, limits=rew_centers, shift_mult=0)

    # ###################################################
    # logger.info("Multiplying hit goal reward by 0.8!")
    # env_train.pd.reward_model.params["hit_goal_reward"] *= 0.8
    # ###################################################

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
        # entropy_cf=LinDecay(1e-2, 50.0, warmup_steps=1_000, trans_steps=5_000),
        # entropy_cf=LinDecay(2e-2, 50.0, warmup_steps=2_000, trans_steps=5_000),
        # entropy_cf=LinDecay(1e-2, 5.0, warmup_steps=2_000, trans_steps=5_000),
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

    env_orig, ppo, _ = load_ppo(ckpt_dir, shift_ranges, rew_centers)

    def pol_expert(obs_priv):
        obs_expert = obs_priv[1:17]
        return ppo.act(obs_expert)

    bc_coeff = 1.0
    make_ppo = ft.partial(make_ppo_ff_bc, pol_expert=pol_expert, bc_coeff=bc_coeff)

    train_ppo(
        wandb_name, key, env_train, env_test, ppo_cfg, collect_cfg, project_name="pdg_ppo_ff_adj_bc", make_ppo=make_ppo
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
