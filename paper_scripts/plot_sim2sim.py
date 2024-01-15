import os
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils.plot_utils import (
    get_action_labels_mathtext_dict,
    get_constr_labels_mathtext_dict,
    setup_nature_style,
)
from pop_down_gym.constants import ShotConstants
from pop_down_gym.physics import shafranov_coeff


@dataclass
class Case:
    df: pd.DataFrame
    name: str
    color: str


def plot_df(cases, constraint_limits, fig_path, plot_name):
    Ip_MA_tgt = 2.0

    plt.style.use("ggplot")
    setup_nature_style()

    ylims = {
        "Ip_MA": [1.2e00, 9.18e00],
        "Bv_dot_mag": [5.14e-02, 3.70e-01],
        "Wdot_mag": [-1.42, 8e01],
        "beta_n": [1.25e-03, 3.00e-02],
        "beta_p": [6.96e-02, 4.57e-01],
        "li": [3.36e-01, 4.0e00],
        "ng_frac": [0.0, 8.20e-01],
        "shafranov_coeff": [1.95, 4.05],
        "iota95": [0.05, 0.28],
    }

    constr_labels_mathtext = get_constr_labels_mathtext_dict()
    constr_vars = constr_labels_mathtext.keys()

    nconstr = len(constr_labels_mathtext.keys())

    width_traj = 6

    figsize = np.array([width_traj, 1.5 * nconstr])
    target_width = 5.35
    figsize *= target_width / figsize[0]
    figsize_main = figsize * np.array([2, 1])

    constr_alpha = 0.25

    fig_main = plt.figure(layout="constrained", figsize=figsize_main)
    subfigs = fig_main.subfigures(1, 2, wspace=0.1)

    # Create the goal + constraint subfigure.
    fig_constr = subfigs[0]
    axs = fig_constr.subplots(nconstr, 1, sharex=True)

    for ax, var in zip(axs, constr_vars):
        for case in cases:
            df = case.df
            (line,) = ax.plot(df.index, df[var], color=case.color)
            ax.set_ylabel(constr_labels_mathtext[var])
            ax.set_ylim(ylims[var])

        if var in constraint_limits.keys():
            ymax = max(ylims[var])
            ax.axhspan(
                min(ymax, constraint_limits[var]),
                ymax,
                fc="C0",
                ec="none",
                alpha=constr_alpha,
            )

        if var == "Ip_MA":
            ax.axhspan(
                min(ylims[var]), Ip_MA_tgt, fc="C5", ec="none", alpha=constr_alpha
            )
        ax.autoscale_view()
    axs[0].set_title("Goal + Constraint Trajectories", fontsize=14)
    axs[-1].set_xlabel("Time (s)")

    action_labels_mathtext = get_action_labels_mathtext_dict()
    action_vars = action_labels_mathtext.keys()

    fig_acts = subfigs[1]
    axs = fig_acts.subplots(len(action_vars), 1, sharex=True)
    for ax, var in zip(axs, action_vars):
        for case in cases:
            df = case.df
            ax.plot(df.index, df[var], color=case.color, label=case.name)
            ax.set_ylabel(action_labels_mathtext[var])
    axs[0].set_title("Action Trajectories", fontsize=14)
    axs[-1].set_xlabel("Time (s)")

    if len(cases) > 1:
        handles, labels = axs[-1].get_legend_handles_labels()
        fig_main.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1))

    fig_main.savefig(fig_path / f"{plot_name}.pdf", bbox_inches="tight")


def process_df(df):
    # Forward fill nans.
    df = df.fillna(method="ffill")

    consts = ShotConstants.for_sparc()

    """
    Calculate derived quantities. Plus rename variables
    to align with the plotting naming schema.
    """

    df["Ip_MA"] = 1e-6 * df["Ip"]
    df["Wdot_mag"] = 1e-6 * np.abs(df["dWtdt"])  # Convert to MW.
    df.attrs["constraint_limits"]["Wdot_mag"] *= 1e-6

    # Differentiate Bv.
    df["Bv_dot_mag"] = np.abs(np.gradient(df["Bv"], df.index))
    df["Ip_dot"] = np.gradient(df["Ip"], df.index)

    # Change betaN from percents to non-dimensional.
    df["beta_n"] = 0.01 * df["betaN"]

    # Compute iota95 from q95
    df["iota95"] = 1.0 / df["q95"]

    # Minor radius = R0 * epsilon
    df["aminor"] = consts.R0 * df["epsilon"]

    # Compute the shafranov coeff.
    df["shafranov_coeff"] = shafranov_coeff(
        consts.R0,
        df["aminor"].to_numpy(),
        df["kappa"].to_numpy(),
        df["betapol"].to_numpy(),
        df["li3"].to_numpy(),
    )

    # Rename things to have a consistent naming scheme.
    df["beta_p"] = df["betapol"]

    df["li"] = df["li3"]

    df["ng_frac"] = df["fne_gr"]

    # Process actions.
    df["dIp_dt"] = 1e-6 * df["dIp_dt"]
    df["dPaux_dt"] = 1e-6 * df["dPaux_dt"]

    constr_labels_mathtext = get_constr_labels_mathtext_dict()
    for k in constr_labels_mathtext.keys():
        assert k in df.columns, f"{k} not in df.columns"

    return df


def plot_es_raptor():
    pkl_path = os.path.join(
        os.path.dirname(__file__), "../tmp/sim2sim_ES_OPENLOOP_DAWSON.pkl"
    )
    df = pd.read_pickle(pkl_path)
    df = process_df(df)
    cases = [Case(df=df, name="ES Openloop", color="C1")]
    plot_df(
        cases,
        df.attrs["constraint_limits"],
        pathlib.Path(pkl_path).parent,
        plot_name="sim2sim_es_openloop",
    )


def plot_baseline_and_ppo():
    ppo_pkl_path = os.path.join(os.path.dirname(__file__), "../tmp/sim2sim_PPO_OSO.pkl")
    df_ppo = process_df(pd.read_pickle(ppo_pkl_path))

    baseline_pkl_path = os.path.join(
        os.path.dirname(__file__), "../tmp/sim2sim_BASELINE.pkl"
    )
    df_baseline = process_df(pd.read_pickle(baseline_pkl_path))

    cases = [
        Case(df=df_baseline, name="Baseline", color="C0"),
        Case(df=df_ppo, name="PPO", color="C1"),
    ]
    plot_df(
        cases,
        df_ppo.attrs["constraint_limits"],
        pathlib.Path(ppo_pkl_path).parent,
        plot_name="sim2sim_ppo_vs_baseline",
    )


if __name__ == "__main__":
    plot_baseline_and_ppo()
    plot_es_raptor()
