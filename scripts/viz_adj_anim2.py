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

    plot_dir = pkl_path.parent
    with open(pkl_path, "rb") as f:
        eval_series = pickle.load(f)

    pkl_path = pkl_path.parent / "fancy_data_offsets.pkl"
    with open(pkl_path, "rb") as f:
        offset_series = pickle.load(f)

    for ll, (b_eval, b_offsets) in enumerate(list(zip(eval_series, offset_series))):
        # if ll < 4:
        #     continue

        datas: list[PPOEval] = b_eval
        anim_T = len(datas)

        constr_labels = PopDownGymStateless.constr_labels()
        nconstr = len(constr_labels)

        dt = 0.05

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

        constr_ub = rew_centers
        Ip_MA_tgt = 2.0

        figsize = np.array([6, 1.2 * nconstr])
        fig, axes = plt.subplots(nconstr, layout="constrained", figsize=figsize, sharex=True, dpi=350)
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

        def init_fn() -> list[plt.Artist]:
            return [*line_cols, *spans]

        def update(kk: int) -> list[plt.Artist]:
            for ii, ax in enumerate(axes):
                label = constr_labels[ii]

                offsets_dict = b_offsets[kk]

                # 1: Udpate line cols.
                bT_rew_input = datas[kk].bT_info["reward_inputs"][label]
                b_segs = get_segs(bT_rew_input, datas[kk].bT_valid_mask, dt)
                line_cols[ii].set_segments(b_segs)

                # 2: Update spans.
                if label in constr_ub:
                    ymin, ymax = ax.get_ylim()
                    constr_ub_ = constr_ub[label]

                    if label in offsets_dict:
                        constr_ub_ = constr_ub_ + offsets_dict[label]

                    ymin_ = min(ymax, constr_ub_)
                    spans[ii].set_xy([[0, ymin_], [1, ymin_], [1, ymax], [0, ymax]])

            return [*line_cols, *spans]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)

        def progress_callback(curr_frame: int, total_frames: int):
            pbar.update(1)

        path = plot_dir / "anim_fancy_{}.mp4".format(ll)

        pbar = tqdm.tqdm(total=anim_T)
        ani.save(path, progress_callback=progress_callback)
        pbar.close()
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
