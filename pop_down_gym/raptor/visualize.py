import numpy as np
import scipy.io as sio
import pandas as pd
from pop_down_gym.pd_gym_stateless import PopDownGymStateless
from pop_down_gym.physics import shafranov_coeff
from pop_down_gym.constants import ShotConstants

def loadmat(filename):
    """https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-
    dictionaries.

    this function should be called instead of direct scipy.io .loadmat as it cures the
    problem of not properly recovering python dictionaries from mat files. It calls the
    function check keys to cure all entries which are still mat-objects
    """

    def _check_keys(dict):
        """Checks if entries in dictionary are mat-objects.

        If yes todict is called to change them to nested dictionaries
        """
        for key in dict:
            if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
                dict[key] = _todict(dict[key])
        return dict

    def _todict(matobj):
        """A recursive function which constructs from matobjects nested dictionaries."""
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def plot_df(df, lines=None, title=None):
    import matplotlib.pyplot as plt

    n_col = len(df.columns)
    # Create subplots for each column
    fig, axes = plt.subplots(
        nrows=len(df.columns), sharex=True, figsize=(10, n_col * 5)
    )

    for col, ax in zip(df.columns, axes):
        df[col].plot(ax=ax, use_index=True)
        ax.set_title(col)
        ax.set_ylabel("Value")
        if lines and col in lines.keys():
            ax.axhline(lines[col], color="r", linestyle="--")

    plt.xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing
    if title:
        plt.suptitle(title)
    plt.show()


if __name__ == "__main__":
    env = PopDownGymStateless.create_env()
    df = pd.read_pickle("sim2sim.pkl")
    consts = ShotConstants.for_sparc()
    df["Wdot_mag"] = np.abs(df["dWtdt"])

    # Differentiate Bv.
    df["Bvdot_mag"] = np.abs(np.gradient(df["Bv"], df.index))
    df["Ip_dot"] = np.gradient(df["Ip"], df.index)

    # Change betaN from percents to non-dimensional.
    df["betaN"] = 0.01 * df["betaN"]

    # Compute iota95 from q95
    df["iota95"] = 1.0/df["q95"]

    # Minor radius = R0 * epsilon
    df["aminor"] = consts.R0 * df["epsilon"]

    # Compute the shafranov coeff.
    df["Gamma"] = shafranov_coeff(consts.R0, df["aminor"].to_numpy(), df["kappa"].to_numpy(), df["betapol"].to_numpy(), df["li3"].to_numpy())

    plot_df(
        df[["Ip", "Bv_dot_mag", "Wdot_mag", "betaN", "betapol", "li3", "fne_gr", "Gamma", "iota95"]],
        df.attrs['constraint_limits'],
        title="RL Controller + Raptor",
    )

    # plot_df(df[["Ip_dot", "Pauxtot", "kappa"]])
