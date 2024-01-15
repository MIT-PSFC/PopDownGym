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

from jaxrl.helpers import get_default_rew_bounds
from jaxrl.ppo import PPOEval
from plot_utils.plot_utils import (get_constr_labels_mathtext, get_constr_labels_mathtext_dict,
                                   get_env_params_mathtext_dict, get_segs, setup_nature_style)
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def main():
    plt.style.use("ggplot")
    setup_nature_style()

    n_rows, n_cols = 2, 4
    figsize = np.array([4.0 * n_rows, 3.0 * n_cols])
    target_width = 5.35
    figsize *= target_width / figsize[0]
    figsize_main = figsize * np.array([2, 1])

    fig_main = plt.figure(layout="constrained", figsize=figsize_main)
    subfigs = fig_main.subfigures(1, 2, wspace=0.1)

    ######################################################################################
    # style_mean = dict(color="C1", zorder=2)
    style_mean = dict(color="C1", zorder=2)
    style_dist = dict(color="C3", alpha=0.3, zorder=1.95)
    # style_q = dict(color="C3", alpha=0.6, zorder=2)

    constr_labels_mathtext = get_constr_labels_mathtext_dict()
    env_params_mathtext = get_env_params_mathtext_dict()

    ######################################################################################
    for jj in range(2):
        fig: plt.SubFigure = subfigs[jj]
        axes2d = fig.subplots(n_cols, n_rows, sharey=True)
        axes = axes2d.flatten()

        # 1: Read data.
        data_dir = pathlib.Path(__file__).parent.parent / "scripts/check_sensitivity/"
        if jj == 0:
            pkl_path = data_dir / "ppo_sens_processed_data.pkl"
            with open(pkl_path, "rb") as f:
                dict_offsets: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = pickle.load(f)
            dict_processed = dict_offsets["mid"]
        else:
            pkl_path = data_dir / "ppo_sens_envparams_processed.pkl"
            with open(pkl_path, "rb") as f:
                dict_processed: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = pickle.load(f)

            axes[-1].set_visible(False)

        for ax, perturb_label in zip(axes, dict_processed.keys()):
            C_vals, C_data = dict_processed[perturb_label]

            if perturb_label == "Wdot_mag":
                C_vals = C_vals * 1e-6

            C_mean, C_mean_lo, C_mean_hi = C_data[:, 0, 0], C_data[:, 0, 1], C_data[:, 0, 2]
            C_q05, C_q05_lo, C_q05_hi = C_data[:, 1, 0], C_data[:, 1, 1], C_data[:, 1, 2]
            C_q95, C_q95_lo, C_q95_hi = C_data[:, 2, 0], C_data[:, 2, 1], C_data[:, 2, 2]

            ax.plot(C_vals, C_mean, **style_mean)
            ax.fill_between(C_vals, C_q05, C_q95, **style_dist)
            # ax.plot(C_vals, C_q05, **style_q)
            # ax.plot(C_vals, C_q95, **style_q)

            if jj == 0:
                ax.set_xlabel(constr_labels_mathtext[perturb_label])
            else:
                ax.set_xlabel(env_params_mathtext[perturb_label])
            ax.set_xlim(C_vals.min(), C_vals.max())

        for ax in axes2d[:, 0]:
            ax.set_ylabel("Time to goal (s)")

        # fig.supylabel("Time to goal (s)")

        panel_letter = ["a", "b"][jj]
        offset = transforms.ScaledTranslation(0 / 72, 0 / 72, fig.dpi_scale_trans)
        trans = transforms.ScaledTranslation(0.0, 1.0, fig.transSubfigure) + offset
        fig.text(0, 0, panel_letter, transform=trans, ha="right", va="bottom", weight="bold", fontsize=14)

    # Legend.
    leg_els = [
        plt.Line2D([0], [0], **style_mean, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, **style_dist, label="95% CI"),
    ]
    legend = fig_main.legend(
        handles=leg_els,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        facecolor="white",
        framealpha=1.0,
        fontsize=12,
    )
    legend.get_frame().set_linewidth(0)

    plot_dir = pathlib.Path(__file__).parent.parent / "tmp/ppo_viz_adj"
    fig_path = plot_dir / "sensitivity.pdf"
    fig_main.savefig(fig_path, bbox_inches="tight", pad_inches=5e-2)
    # fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", pad_inches=5e-2)
    plt.close(fig_main)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
