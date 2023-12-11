import click
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from pop_down_gym.physics import shafranov_coeff
from pop_down_gym.constants import ShotConstants
from plot_utils.plot_utils import get_constr_labels_mathtext_dict, get_action_labels_mathtext_dict, setup_nature_style


def plot_df(df, constraint_limits, fig_path, title=None):

    Ip_MA_tgt = 2.0

    plt.style.use("ggplot")
    setup_nature_style()

    ylims = {
        "Ip_MA": [1.2e00, 9.18e00],
        "Bv_dot_mag": [5.14e-02, 3.70e-01],
        "Wdot_mag": [-1.42e06, 8e07],
        "beta_n": [1.25e-03, 3.00e-02],
        "beta_p": [6.96e-02, 4.57e-01],
        "li": [3.36e-01, 3.10e00],
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
        ax.plot(df.index, df[var], color="C1")
        ax.set_ylabel(constr_labels_mathtext[var])
        ax.set_ylim(ylims[var])
        if var in constraint_limits.keys():
            ymax = max(ylims[var])
            ax.axhspan(min(ymax, constraint_limits[var]), ymax, fc="C0", ec="none", alpha=constr_alpha)

        if var == "Ip_MA":
            ax.axhspan(min(ylims[var]), Ip_MA_tgt, fc="C5", ec="none", alpha=constr_alpha)
        ax.autoscale_view()
    axs[0].set_title("Constraint Trajectories", fontsize=14)
    axs[-1].set_xlabel("Time (s)")

    action_labels_mathtext = get_action_labels_mathtext_dict()
    action_vars = action_labels_mathtext.keys()
    
    fig_acts = subfigs[1]
    axs = fig_acts.subplots(len(action_vars), 1, sharex=True)
    for ax, var in zip(axs, action_vars):
        ax.plot(df.index, df[var])
        ax.set_ylabel(action_labels_mathtext[var])
    axs[0].set_title("Action Trajectories", fontsize=14)
    axs[-1].set_xlabel("Time (s)")
    fig_main.savefig(fig_path / "sim2sim.pdf", bbox_inches="tight")
    plt.show()

def process_df(df):

    # Forward fill nans.
    df = df.fillna(method="ffill")

    consts = ShotConstants.for_sparc()


    """
    Calculate derived quantities. Plus rename variables
    to align with the plotting naming schema.
    """

    df["Ip_MA"] = 1e-6 * df["Ip"]
    df["Wdot_mag"] = np.abs(df["dWtdt"])

    # Differentiate Bv.
    df["Bv_dot_mag"] = np.abs(np.gradient(df["Bv"], df.index))
    df["Ip_dot"] = np.gradient(df["Ip"], df.index)

    # Change betaN from percents to non-dimensional.
    df["beta_n"] = 0.01 * df["betaN"]

    # Compute iota95 from q95
    df["iota95"] = 1.0/df["q95"]

    # Minor radius = R0 * epsilon
    df["aminor"] = consts.R0 * df["epsilon"]

    # Compute the shafranov coeff.
    df["shafranov_coeff"] = shafranov_coeff(consts.R0, df["aminor"].to_numpy(), df["kappa"].to_numpy(), df["betapol"].to_numpy(), df["li3"].to_numpy())

    # Rename things to have a consistent naming scheme.
    df["beta_p"] = df["betapol"]

    df["li"] = df["li3"]

    df["ng_frac"] = df["fne_gr"]

    # Process actions.
    df['dIp_dt'] = 1e-6 * df['dIp_dt']
    df['dPaux_dt'] = 1e-6 * df['dPaux_dt']


    constr_labels_mathtext = get_constr_labels_mathtext_dict()
    for k in constr_labels_mathtext.keys():
        assert k in df.columns, f"{k} not in df.columns"
    
    return df

@click.command()
@click.option("--pkl_path", type=str, default=os.path.join(os.path.dirname(__file__), "../tmp/sim2sim_PPO_OSO.pkl"))
def main(pkl_path):
    pkl_path = pathlib.Path(pkl_path)
    df = pd.read_pickle(pkl_path)
    df = process_df(df)
    plot_df(df, df.attrs['constraint_limits'], pkl_path.parent, title="RL Controller + Raptor")
    
if __name__ == "__main__":
    main()