import pathlib
import pickle

import ipdb
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import orbax
import typer
from loguru import logger

from jaxrl.env_types import BTControl, TControl
from jaxrl.ppo import Collector, CollectorCfg, PPOAlg, PPOCfg, PPOEval
from jaxrl.utils.ckpt_manager import get_ckpt_manager_sync
from jaxrl.utils.jax_types import BTBool, BTFloat
from jaxrl.utils.jax_utils import jax2np
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj


def get_ppo(ckpt_dir: pathlib.Path, env: PDEnvAdj):
    ppo_cfg = PPOCfg(
        pol_lr=3e-4,
        val_lr=3e-4,
        entropy_cf=1.0,
        disc_gamma=0.99,
        pol_hid_sizes=[256, 256, 256],
        val_hid_sizes=[256, 256, 256],
        act="tanh",
        pol_type="TanhNormal",
        train_cfg=None,
        rew_scale=5e2,
        clip_grad=1.0,
    )
    ppo = PPOAlg.create(jr.PRNGKey(5123), env, ppo_cfg)

    if (ckpt_dir / "checkpoint").exists() and (ckpt_dir / "checkpoint").is_file():
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ppo_dict = orbax_checkpointer.restore(ckpt_dir, item={"ppo": ppo})
        ppo: PPOAlg = ppo_dict["ppo"]
        plot_dir = ckpt_dir.parent / "ppo_viz_adj"
    else:
        ckpt_manager = get_ckpt_manager_sync(ckpt_dir, max_to_keep=100)
        step = ckpt_manager.latest_step()
        ppo_dict = ckpt_manager.restore(step, items={"ppo": ppo})
        ppo: PPOAlg = ppo_dict["ppo"]

        plot_dir = ckpt_dir.parent / "eval_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    return ppo, plot_dir


def get_utraj_mean(bT_valid: BTBool, bT_control: BTControl) -> TControl:
    T_num_valid = np.sum(bT_valid, axis=0)
    bT_control_valid = np.where(bT_valid[:, :, None], bT_control, 0)
    T_control_mean = np.sum(bT_control_valid, axis=0) / T_num_valid[:, None]
    return T_control_mean


def get_utraj_quantile(bT_valid: BTBool, bT_rew: BTFloat, bT_control: BTControl, q: float) -> TControl:
    # High q means risk-averse (i.e., low quantiles of the reward).
    q = 1 - q
    b = len(bT_valid)

    b_rew_sum = np.where(bT_valid, bT_rew, 0).sum(axis=1)

    # 0 => lowest reward, 1 => highest reward.
    b_rew_idx = np.argsort(b_rew_sum, axis=0)
    idx_lo = np.floor(q * b).astype(int)
    idx_hi = np.ceil(q * b).astype(int)

    # Take the controls corresponding to the q-th quantile of the reward.
    T_controls_lo = bT_control[b_rew_idx[idx_lo], :, :]
    T_controls_hi = bT_control[b_rew_idx[idx_hi], :, :]
    T_control = 0.5 * (T_controls_lo + T_controls_hi)

    return T_control


def get_utraj_cvar(bT_valid: BTBool, bT_rew: BTFloat, bT_control: BTControl, q: float) -> TControl:
    # High q means risk-averse (i.e., low quantiles of the reward).
    q = 1 - q

    b_rew_sum = np.where(bT_valid, bT_rew, 0).sum(axis=1)

    # 0 => lowest reward, 1 => highest reward.
    b_rew_idx = np.argsort(b_rew_sum, axis=0)
    idx = np.ceil(q * len(bT_valid)).astype(int)

    # If its invalid, set it to 0. When we sum it, it gets ignored.
    bT_control_valid = np.where(bT_valid[:, :, None], bT_control, 0)
    bT_control_sort = bT_control_valid[b_rew_idx, :]
    bT_valid_sort = bT_valid[b_rew_idx, :]
    T_num_valid = np.sum(bT_valid_sort[:idx, :], axis=0)
    T_control = np.sum(bT_control_sort[:idx, :], axis=0) / T_num_valid[:, None]

    return T_control


def main(ckpt_dir: pathlib.Path):
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
    env = PDEnvAdj(shift_ranges=shift_ranges, limits=rew_centers, shift_mult=0)
    ppo, plot_dir = get_ppo(ckpt_dir, env)
    ######################################################################################
    collect_cfg = CollectorCfg(0, 0, n_env_eval=256, rollout_T_eval=120)
    collector = Collector.create(jr.PRNGKey(1234), env, collect_cfg)

    # 1: Do a rollout with the PPO policy and compute the distribution of rewards and controls (due to env).
    logger.info("Rolling out PPO...")
    data_eval: PPOEval = jax2np(ppo.eval(collector))

    bT_u_mean = get_utraj_mean(data_eval.bT_valid_mask, data_eval.bT_control)
    bT_u_q50 = get_utraj_quantile(data_eval.bT_valid_mask, data_eval.bT_rew, data_eval.bT_control, q=0.50)
    bT_u_q95 = get_utraj_quantile(data_eval.bT_valid_mask, data_eval.bT_rew, data_eval.bT_control, q=0.95)
    bT_u_cvar50 = get_utraj_cvar(data_eval.bT_valid_mask, data_eval.bT_rew, data_eval.bT_control, q=0.50)
    bT_u_cvar95 = get_utraj_cvar(data_eval.bT_valid_mask, data_eval.bT_rew, data_eval.bT_control, q=0.95)
    bT_u_cvar99 = get_utraj_cvar(data_eval.bT_valid_mask, data_eval.bT_rew, data_eval.bT_control, q=0.99)

    bT_controls = {
        "mean": bT_u_mean,
        "q50": bT_u_q50,
        "q95": bT_u_q95,
        "cvar50": bT_u_cvar50,
        "cvar95": bT_u_cvar95,
        "cvar99": bT_u_cvar99,
    }

    @jax.jit
    def collect_ff(T_control_: TControl):
        bT_state, bT_out = collector.rollout_ff(T_control_)

        # Compute the mask.
        batch_size = len(bT_out.terminated)
        bT_invalid = lax.cummax(1 * bT_out.terminated, axis=1) > 0
        bT_valid_mask = jnp.concatenate([jnp.ones((batch_size, 1), dtype=bool), ~bT_invalid[:, :-1]], axis=1)
        assert bT_valid_mask.dtype == bool

        return bT_valid_mask, bT_state, T_control_, bT_out.reward, bT_out.info

    ppo_data = data_eval.bT_valid_mask, data_eval.bT_state, data_eval.bT_control, data_eval.bT_rew, data_eval.bT_info
    data = {"ppo": ppo_data}

    for k, T_control in bT_controls.items():
        logger.info(f"Rollout out {k}...")
        data[k] = jax2np(collect_ff(T_control))

    # Save data.
    pkl_path = plot_dir / "compare_ppo_ff_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved data to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
