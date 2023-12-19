import ipdb
import jax.random as jr
import typer
from loguru import logger

from jaxrl.ppo import CollectorCfg, PPOCfg, PPOTrainCfg
from jaxrl.ppo_trainer import train_ppo
from jaxrl.utils.logging import set_logger_format
from jaxrl.utils.schedule import LinDecay
from pop_down_gym.pd_gym_jaxrl import PDEnvFFAdj
from pop_down_gym.ppo_ff import make_ppo_ff


def main(wandb_name: str = None):
    set_logger_format()

    key = jr.PRNGKey(54123)

    rew_bounds = {
        "li": [2, 3],
        "ng_frac": [0.5, 0.8],
        "beta_n": [0.015, 0.028],
        "beta_p": [0.25, 0.4],
        "Bv_dot_mag": [0.2, 0.4],
        "Wdot_mag": [20_000_000, 70_000_000],
        "shafranov_coeff": [3.4, 3.6],
        "iota95": [0.35, 0.45],
    }
    rew_centers = {k: 0.5 * (v[0] + v[1]) for k, v in rew_bounds.items()}
    shift_ranges = {k: 0.5 * (v[1] - v[0]) for k, v in rew_bounds.items()}

    logger.info("Constructing Env...")
    env_train = PDEnvFFAdj(shift_ranges=shift_ranges, limits=rew_centers)
    env_test = PDEnvFFAdj(shift_ranges=shift_ranges, limits=rew_centers, shift_mult=0)

    ###################################################
    logger.info("Multiplying hit goal reward by 0.25!")
    env_train.pd.reward_model.params["hit_goal_reward"] *= 0.25
    logger.info("Setting min_reward to -0.25")
    env_train.pd.reward_model.ip_ma["min_reward"] = -0.25
    ###################################################

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

    n_iters = 25_000
    train_ppo(
        wandb_name,
        key,
        env_train,
        env_test,
        ppo_cfg,
        collect_cfg,
        project_name="pdg_ppo_ff_adj",
        make_ppo=make_ppo_ff,
        n_iters=n_iters,
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
