import copy
import pathlib
import pickle

import ipdb
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import orbax
import tqdm
import typer
from loguru import logger
from matplotlib.collections import LineCollection

from jaxrl.env_types import BTControl, BTState, TControl
from jaxrl.ppo import Collector, CollectorCfg, PPOAlg, PPOCfg, PPOEval
from jaxrl.utils.ckpt_manager import get_ckpt_manager_sync
from jaxrl.utils.jax_types import BTBool, BTFloat
from jaxrl.utils.jax_utils import jax2np
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj
from pop_down_gym.pd_gym_stateless import PopDownGymStateless

EvalData = tuple[BTBool, BTState, BTControl, BTFloat, dict]


def get_segs(bT_rew, bT_valid_mask, dt: float):
    b_segs = []
    for bb, T_r in enumerate(bT_rew):
        # Truncate to valid only.
        if not np.all(bT_valid_mask[bb]):
            idx_first_invalid = bT_valid_mask[bb].argmin()
            T_r = T_r[:idx_first_invalid]

        T = len(T_r)
        T_ts = dt * np.arange(T)
        # (T, 2)
        segs = np.stack([T_ts, T_r], axis=1)
        b_segs.append(segs)
    return b_segs


def main(pkl_path: pathlib.Path):
    logger.info(f"Loading from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data_dict: dict[str, EvalData] = pickle.load(f)
    logger.info("Plotting...")
    n_runs = len(data_dict)

    dt = 0.05
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
    constr_ub = rew_centers
    Ip_MA_tgt = 2.0

    bT_valid_ppo = data_dict["ppo"][0]
    T_ppo = np.any(bT_valid_ppo, axis=0).sum()

    ylims = {
        "Ip_MA": [1.2e00, 9.18e00],
        "Bv_dot_mag": [5.14e-02, 3.30e-01],
        "Wdot_mag": [-1.42e06, 4.93e07],
        "beta_n": [1.25e-03, 1.17e-02],
        "beta_p": [6.96e-02, 4.57e-01],
        "li": [3.36e-01, 4.19e00],
        "ng_frac": [2.30e-01, 7.00e-01],
        "shafranov_coeff": [1.95, 4.05],
        "iota95": [0.05, 0.25],
    }

    # Plot the constraints.
    constr_labels = PopDownGymStateless.constr_labels()
    nconstr = len(constr_labels)
    ncontrol =

    figsize = np.array([4 * n_runs, 1.2 * nconstr])
    fig, axes = plt.subplots(nconstr, n_runs, figsize=figsize, layout="constrained", sharex=True)

    for jj, (k, data) in enumerate(data_dict.items()):
        bT_valid_mask, bT_state, bT_control, bT_rew, bT_info = data
        bT_rew_inputs = bT_info["reward_inputs"]

        axes[0, jj].set_title(k)

        for ii, ax in enumerate(axes[:, jj]):
            label = constr_labels[ii]

            bT_rew_input = bT_rew_inputs[label]
            b_segs = get_segs(bT_rew_input, bT_valid_mask, dt)
            col = LineCollection(b_segs, color="C1", lw=0.5, alpha=0.4)
            ax.add_collection(col)

            if jj == 0:
                ax.set_ylabel(label, rotation=0, ha="right")

    # Plot the limits.
    for jj, (k, data) in enumerate(data_dict.items()):
        for ii, ax in enumerate(axes[:, jj]):
            label = constr_labels[ii]
            ax.set_ylim(ylims[label])
            ax.set_xlim(-0.02, T_ppo * dt + 0.02)
            ymin, ymax = ax.get_ylim()

            if label in constr_ub:
                # Expand the ymax a bit.
                yrange = ymax - ymin
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
                ymin, ymax = ax.get_ylim()

                # if ymax > constr_ub[label]:
                ax.axhspan(min(ymax, constr_ub[label]), ymax, color="C0", alpha=0.2)

            if label == "Ip_MA":
                ax.axhspan(ymin, Ip_MA_tgt, color="C5", alpha=0.2)

    fig_path = "compare_ff.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    logger.info("Saved to {}!".format(fig_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
