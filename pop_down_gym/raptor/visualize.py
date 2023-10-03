import glob

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
from tqdm import tqdm
import pandas as pd


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


def convert_to_df(data):
    time = data["time"]

    def maybe_take_edge(var):
        var_data = data[var]
        if var_data.ndim == 2:
            return var_data[-1, :]
        elif var_data.ndim == 1:
            return var_data
        else:
            raise ValueError("var_data.ndim = {}".format(var_data.ndim))
        return

    df_dic = {}
    for key in data.keys():
        if (
            key == "time"
            or not isinstance(data[key], np.ndarray)
            or (data[key].ndim != 1 and data[key].ndim != 2)
            or time.size not in data[key].shape
        ):
            continue
        df_dic[key] = maybe_take_edge(key)

    df = pd.DataFrame(df_dic, index=time)
    return df


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
    data = loadmat("best_result.mat")
    df = convert_to_df(data["out"])
    import pdb

    pdb.set_trace()
    df["Wdot_mag"] = np.abs(df["dWtdt"])

    # Differentiate Bv.
    df["Bvdot_mag"] = np.abs(np.gradient(df["Bv"], df.index))
    df["Ip_dot"] = np.gradient(df["Ip"], df.index)
    lines = {
        "Bvdot_mag": 0.3,
        "Wdot_mag": 4e7,
        "betaN": 1.5,
        "betapol": 0.3,
        "li3": 3.0,
        "fne_gr": 0.5,
    }
    plot_df(
        df[["Ip", "Bvdot_mag", "Wdot_mag", "betaN", "betapol", "li3", "fne_gr"]],
        lines,
        title="RL Controller + Raptor",
    )

    plot_df(df[["Ip_dot", "Pauxtot", "kappa"]])

    import pdb

    pdb.set_trace()
