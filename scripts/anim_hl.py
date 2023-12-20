import copy
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import typer
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from jaxrl.helpers import get_default_rew_bounds, load_ppo
from jaxrl.ppo import Collector, CollectorCfg, PPOEval
from jaxrl.utils.jax_utils import jax2np
from plot_utils.plot_utils import get_segs
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def main(pkl_path: pathlib.Path):
    with open(pkl_path, "rb") as f:
        eval_datas: dict[float, PPOEval] = pickle.load(f)
    ####################################################################################################
    rew_centers, shift_ranges, rew_min, rew_max = get_default_rew_bounds()
    anim_T = len(eval_datas)

    constr_ub = rew_centers
    constr_ub["Wdot_mag"] = rew_centers["Wdot_mag"] - shift_ranges["Wdot_mag"]
    Ip_MA_tgt = 2.0

    hl_factors = list(eval_datas.keys())
    datas = list(eval_datas.values())

    ylims = {
        "Ip_MA": [1.2e00, 9.18e00],
        "Bv_dot_mag": [5.14e-02, 3.30e-01],
        "Wdot_mag": [-1.42e06, 4.93e07],
        "beta_n": [1.25e-03, 1.17e-02],
        "beta_p": [6.96e-02, 4.57e-01],
        "li": [3.36e-01, 4.19e00],
        "ng_frac": [2.30e-01, 8.20e-01],
        "shafranov_coeff": [1.95, 4.05],
        "iota95": [0.05, 0.25],
    }

    # KbT_Wdot_mags = np.stack([datas[kk].bT_info["reward_inputs"]["Wdot_mag"] for kk in range(anim_T)], axis=0)
    # ipdb.set_trace()

    dt = 0.05

    constr_labels = PopDownGymStateless.constr_labels()
    nconstr = len(constr_labels)

    figsize = np.array([6, 1.2 * nconstr])
    # dpi = 350
    dpi = 200
    fig, axes = plt.subplots(nconstr, layout="constrained", figsize=figsize, sharex=True, dpi=dpi)
    line_cols, spans = [], []
    ax: plt.Axes
    for ii, ax in enumerate(axes):
        label = constr_labels[ii]

        bT_rew_input = datas[0].bT_info["reward_inputs"][label]
        b_segs = get_segs(bT_rew_input, datas[0].bT_valid_mask, dt)
        col = LineCollection(b_segs, color="C1", lw=0.5, alpha=0.4)
        line_cols.append(col)
        ax.add_collection(col)
        ax.set_ylabel(label, rotation=0, ha="right")

        # Set the limits.
        ax.autoscale_view()
        ax.set_ylim(ylims[label])

        # Plot the limits.
        ymin, ymax = ax.get_ylim()
        if label in constr_ub:
            # Expand the ymax a bit.
            yrange = ymax - ymin
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            ymin, ymax = ax.get_ylim()

            # if ymax > constr_ub[label]:
            rect = ax.axhspan(min(ymax, constr_ub[label]), ymax, color="C0", alpha=0.2)
            spans.append(rect)
        else:
            spans.append(plt.Line2D([], []))

        if label == "Ip_MA":
            ax.axhspan(ymin, Ip_MA_tgt, color="C5", alpha=0.2)

    hl_text = axes[0].set_title("hl_factor: {}".format(0))

    def init_fn() -> list[plt.Artist]:
        return [*line_cols, *spans, hl_text]

    def update(kk: int) -> list[plt.Artist]:
        for ii, ax in enumerate(axes):
            label = constr_labels[ii]

            # 1: Udpate line cols.
            bT_rew_input = datas[kk].bT_info["reward_inputs"][label]
            b_segs = get_segs(bT_rew_input, datas[kk].bT_valid_mask, dt)
            line_cols[ii].set_segments(b_segs)

            # 2: Update spans.
            if label in constr_ub:
                ymin, ymax = ax.get_ylim()
                constr_ub_ = constr_ub[label]
                ymin_ = min(ymax, constr_ub_)
                spans[ii].set_xy([[0, ymin_], [1, ymin_], [1, ymax], [0, ymax]])

        hl_text.set_text("hl_factor: {:.3f}".format(hl_factors[kk]))

        return [*line_cols, *spans, hl_text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(1)

    # plot_dir = pkl_path.parent
    # path = plot_dir / "anim_hlfactor.mp4"
    path = pkl_path.with_suffix(".mp4")

    pbar = tqdm.tqdm(total=anim_T)
    ani.save(path, progress_callback=progress_callback)
    pbar.close()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
