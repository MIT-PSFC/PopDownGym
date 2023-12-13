import pathlib
import pickle

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import typer
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from jaxrl.ppo import PPOEval
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def main(pkl_path: pathlib.Path):
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

    with open(pkl_path, "rb") as f:
        offset_series = pickle.load(f)

    plot_dir = pkl_path.parent

    # constr_labels = PopDownGymStateless.constr_labels()
    rew_labels = list(rew_bounds.keys())
    nconstr = len(rew_bounds)

    #######################################################################################
    for ll, b_offsets in enumerate(offset_series):
        anim_T = len(b_offsets)

        figsize = np.array([1.2 * nconstr, 3.0])
        fig, axes = plt.subplots(1, nconstr, layout="constrained", figsize=figsize, sharex=True, dpi=300)

        rects = []
        for ii, ax in enumerate(axes):
            rew_lb, rew_ub = rew_bounds[rew_labels[ii]]
            ax.set_ylim(rew_lb, rew_ub)
            ax.set_xlim(0, 1)
            ax.set_title(rew_labels[ii], fontsize=9, y=-0.15)

            ax.set_facecolor("white")
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            [i.set_visible(True) for i in ax.spines.values()]
            [i.set(color="0.3") for i in ax.spines.values()]
            [i.set_linewidth(0.8) for i in ax.spines.values()]

            height = (rew_ub - rew_lb) / 2
            rect = plt.Rectangle((0, rew_ub), 1, -height, color="C0", alpha=0.6)
            ax.add_patch(rect)
            rects.append(rect)

        fig.get_layout_engine().set(w_pad=0.2, h_pad=4 / 72, hspace=0, wspace=0)

        def init_fn():
            return rects

        def update(kk: int):
            for ii, rect in enumerate(rects):
                offset = b_offsets[kk][rew_labels[ii]]
                val = rew_centers[rew_labels[ii]] + offset
                height = val - rew_bounds[rew_labels[ii]][1]
                rect.set_height(height)
            return rects

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)

        def progress_callback(curr_frame: int, total_frames: int):
            pbar.update(1)

        path = plot_dir / "anim_fancy_bar_{}.mp4".format(ll)

        pbar = tqdm.tqdm(total=anim_T)
        ani.save(path, progress_callback=progress_callback)
        pbar.close()
        plt.close(fig)

        # bars = []
        # for ax in axes:
        #     rect = plt.Rectangle()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
