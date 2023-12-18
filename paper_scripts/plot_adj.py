import pathlib
import pickle

import ipdb
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import tqdm
import typer
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from jaxrl.default_rew_bounds import get_default_rew_bounds
from jaxrl.ppo import PPOEval
from plot_utils.plot_utils import get_constr_labels_mathtext, get_segs, setup_nature_style
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def main(pkl_path: pathlib.Path):
    rew_centers, shift_ranges, rew_min, rew_max = get_default_rew_bounds()

    constr_ub = rew_centers
    Ip_MA_tgt = 2.0

    plot_dir = pkl_path.parent

    setup_nature_style()

    with open(pkl_path, "rb") as f:
        eval_datas, interp_fracs = pickle.load(f)
    datas: list[PPOEval] = eval_datas

    ylims = {
        "Ip_MA": [1.2e00, 9.18e00],
        "Bv_dot_mag": [5.14e-02, 3.30e-01],
        "Wdot_mag": [-1.42e06, 4.93e07],
        "beta_n": [1.25e-03, 1.17e-02],
        "beta_p": [6.96e-02, 4.57e-01],
        "li": [3.36e-01, 4.19e00],
        "ng_frac": [2.30e-01, 8.20e-01],
        "shafranov_coeff": [1.95, 4.05],
        "iota95": [0.05, 0.28],
    }

    dt = 0.05

    constr_labels = PopDownGymStateless.constr_labels()
    constr_labels_mathtext = get_constr_labels_mathtext()
    assert len(constr_labels) == len(constr_labels_mathtext)
    nconstr = len(constr_labels)

    width_traj = 6
    width_constr = 0.5
    width_total = width_traj + width_constr
    width_ratios = [width_traj, width_constr]
    figsize = np.array([width_total, 1.2 * nconstr])
    target_width = 5.35
    figsize *= target_width / figsize[0]
    fig, axes = plt.subplots(nconstr, 2, figsize=figsize, width_ratios=width_ratios, layout="constrained", sharex="col")

    fig.get_layout_engine().set(wspace=0.1)

    # Hide the constraint ax for li.
    axes[0, 1].set_visible(False)

    constr_alpha = 0.25

    data = datas[0]

    # Get the number of timesteps of the traj that lasts the longest.
    T_max = np.argmin(data.bT_valid_mask, axis=1).max()

    # Plot trajs.
    for ii, ax in enumerate(axes[:, 0]):
        label = constr_labels[ii]

        bT_rew_input = data.bT_info["reward_inputs"][label]
        b_segs = get_segs(bT_rew_input, data.bT_valid_mask, dt)
        col = LineCollection(b_segs, color="C1", lw=0.5, alpha=0.4)
        ax.add_collection(col)
        ax.set_ylabel(constr_labels_mathtext[ii])

        ax.autoscale_view()
        ax.set_ylim(ylims[label])

        # Plot the limits.
        ymin, ymax = ax.get_ylim()
        if label in constr_ub:
            # # Expand the ymax a bit.
            # yrange = ymax - ymin
            # ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            # ymin, ymax = ax.get_ylim()
            ax.axhspan(min(ymax, constr_ub[label]), ymax, fc="C0", ec="none", alpha=constr_alpha)

        if label == "Ip_MA":
            ax.axhspan(ymin, Ip_MA_tgt, fc="C5", ec="none", alpha=constr_alpha)

    # Visualize the constraint.
    for ii, ax in enumerate(axes[:, 1]):
        label = constr_labels[ii]
        if label == "Ip_MA":
            continue

        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        [i.set_visible(True) for i in ax.spines.values()]
        ax.set_ylim(rew_min[label], rew_max[label])
        ax.set_xlim(0, 1)

        val = constr_ub[label]
        rew_ub = rew_max[label]
        rect = plt.Rectangle((0, val), 1, rew_ub - val, color="C0", alpha=constr_alpha)
        ax.add_patch(rect)

    # axes[1, 1].set_title("Conditioned\nConstraints", fontsize=12)
    title = "Conditioned\nConstraints"
    offset = transforms.ScaledTranslation(-4 / 72, 7 / 72, fig.dpi_scale_trans)
    trans = transforms.ScaledTranslation(0.5, 1.0, axes[1, 1].transAxes) + offset
    fig.text(0.0, 0.0, title, transform=trans, ha="center", va="bottom", color="#555555", fontsize=12)

    axes[-1, 0].set_xlim(0, T_max * dt)
    txt: plt.Text = axes[-1, 0].set_xlabel("Time (s)")
    print("fontsize: ", txt.get_fontsize())
    fig.align_ylabels(axes[:, 0])
    fig.savefig(plot_dir / "test_style.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
