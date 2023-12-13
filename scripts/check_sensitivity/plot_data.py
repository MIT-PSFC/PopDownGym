import pathlib
import pickle

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats

from jaxrl.utils.jax_types import BBool, BInt
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def main():
    # 1: Read data.
    pkl_path = pathlib.Path(__file__).parent / "ppo_sens_processed_data.pkl"
    with open(pkl_path, "rb") as f:
        dict_offsets: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = pickle.load(f)

    for offset_type, dict_processed in dict_offsets.items():
        # C_vals: (n_constr, )
        # C_data: (n_constr, [mean, q05, q95], [val, ci_lo, ci_hi] )
        # C_vals, C_data = dict_processed["li"]
        # print(C_data.shape)

        rew_labels = list(dict_processed.keys())
        n_rew = len(rew_labels)
        assert n_rew == 8
        n_rows, n_cols = 2, 4

        figsize = 0.8 * np.array([n_cols * 3, n_rows * 2])
        fig, axes2d = plt.subplots(n_rows, n_cols, figsize=figsize, layout="constrained", sharey=True, dpi=400)
        axes = axes2d.flatten()

        style_mean = dict(color="C1", zorder=2)
        style_dist = dict(fc="C3", ec="none", alpha=0.3, zorder=1.95)
        style_q = dict(color="C3", alpha=0.6, zorder=2)
        style_q_err = dict()

        # # Make the y axis the same on all.
        # q05_los = min([C_data[:, 1, 1].min() for _, C_data in dict_processed.values()])
        # q95_his = max([C_data[:, 2, 2].min() for _, C_data in dict_processed.values()])

        for ii, ax in enumerate(axes):
            rew_label = rew_labels[ii]
            C_vals, C_data = dict_processed[rew_label]

            C_mean, C_mean_lo, C_mean_hi = C_data[:, 0, 0], C_data[:, 0, 1], C_data[:, 0, 2]
            C_q05, C_q05_lo, C_q05_hi = C_data[:, 1, 0], C_data[:, 1, 1], C_data[:, 1, 2]
            C_q95, C_q95_lo, C_q95_hi = C_data[:, 2, 0], C_data[:, 2, 1], C_data[:, 2, 2]

            C_q05_err = np.stack([C_q05 - C_q05_lo, C_q05_hi - C_q05], axis=0)
            C_q95_err = np.stack([C_q95 - C_q95_lo, C_q95_hi - C_q95], axis=0)

            ax.plot(C_vals, C_mean, **style_mean)
            ax.fill_between(C_vals, C_q05, C_q95, **style_dist)
            ax.plot(C_vals, C_q05, **style_q)
            ax.plot(C_vals, C_q95, **style_q)
            # ax.errorbar(C_vals, C_q05, yerr=C_q05_err, **style_q_err)
            # ax.errorbar(C_vals, C_q95, yerr=C_q95_err, **style_q_err)
            ax.set_xlabel(rew_label)
            ax.set_xlim(C_vals.min(), C_vals.max())

            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.set_facecolor("#F0F0F0")

        # [ax.set_ylabel("Time to goal (s)") for ax in axes2d[:, 0]]
        fig.supylabel("Time to goal (s)")

        # Custom legend.
        leg_els = [
            plt.Line2D([0], [0], **style_mean, label="Mean"),
            plt.Rectangle((0, 0), 1, 1, **style_dist, label="90% CI"),
        ]
        fig.legend(handles=leg_els, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.98))

        fig_path = pathlib.Path(__file__).parent / "ppo_sens_{}.pdf".format(offset_type)
        fig.savefig(fig_path, bbox_inches="tight", pad_inches=5e-2)
        fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", pad_inches=5e-2)
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
