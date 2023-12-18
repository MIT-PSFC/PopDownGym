import copy
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import orbax
import tqdm
import typer
from loguru import logger
from matplotlib.collections import LineCollection

from jaxrl.ppo import Collector, CollectorCfg, PPOAlg, PPOCfg
from jaxrl.utils.ckpt_manager import get_ckpt_manager_sync
from jaxrl.utils.jax_utils import jax2np
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj


def get_default_rew_bounds():
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
    rew_min = {k: v[0] for k, v in rew_bounds.items()}
    rew_max = {k: v[1] for k, v in rew_bounds.items()}

    return rew_centers, shift_ranges, rew_min, rew_max


def get_constr_vals_from_interp(interp_frac: np.ndarray, shift_ranges: dict, rew_centers: dict):
    offset_dict = {}
    val_dict = {}
    for ii, (key, shift) in enumerate(shift_ranges.items()):
        offset_dict[key] = interp_frac[ii] * shift
        val_dict[key] = rew_centers[key] + offset_dict[key]

    return offset_dict, val_dict


def load_ppo(ckpt_dir: pathlib.Path, shift_ranges, rew_centers):
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

    env = PDEnvAdj(shift_ranges=shift_ranges, limits=rew_centers, shift_mult=0)
    ppo = PPOAlg.create(jr.PRNGKey(5123), env, ppo_cfg)

    if (ckpt_dir / "checkpoint").exists() and (ckpt_dir / "checkpoint").is_file():
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ppo_dict = orbax_checkpointer.restore(ckpt_dir, item={"ppo": ppo})
        ppo: PPOAlg = ppo_dict["ppo"]
        step = int(ppo.update_idx)

        plot_dir = ckpt_dir.parent / "ppo_viz_adj"
    else:
        ckpt_manager = get_ckpt_manager_sync(ckpt_dir, max_to_keep=100)
        step = ckpt_manager.latest_step()
        ppo_dict = ckpt_manager.restore(step, items={"ppo": ppo})
        ppo: PPOAlg = ppo_dict["ppo"]

        plot_dir = ckpt_dir.parent / "eval_plots"

    return env, ppo, plot_dir
