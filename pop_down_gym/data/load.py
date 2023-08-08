import os
import xarray as xr
import numpy as np
import pop_down_gym
import pop_down_gym.constants as constants

MU0 = 4 * np.pi * 1e-7  # [H/m] or [N/A^2]
R0 = 1.85  # Major radius [m]

def apply_episodewise_function(ds: xr.Dataset, func):
    """Apply a function to each episode in a dataset.

    Args:
        ds (xr.Dataset): Dataset with episodes as a dimension.
        func (function): Function to apply to each episode.

    Returns:
        xr.Dataset: Dataset with the function applied to each episode.
    """
    return ds.groupby("episode").apply(
        lambda eps_data: func(eps_data.dropna(dim="time", how="all"))
    )


def vind_24(eps_df):
    # Equation 24 in Romero.
    out = (1.0 / eps_df["Ip"]) * (0.5 * eps_df["Li"] * eps_df["Ip"] ** 2.0).differentiate("time")
    return out

def preprocess(dataset: xr.Dataset, ngauss: int, downsample_fact: int=10, shot_constants=constants.ShotConstants.for_sparc()):
    dataset_stacked = dataset.stack(time_slices=["episode", "time"]).dropna(dim="time_slices")
    dataset_stacked = dataset_stacked.isel(time_slices=slice(None, None, downsample_fact))

    # Total auxiliary power.
    dataset_stacked["Paux_MW"] = 1e-6 * (dataset_stacked["Pauxe"] + dataset_stacked["Pauxi"])

    # Treat Zeff as line average of "ze" profile, which from RAPTOR should be constant?
    dataset_stacked["Zeff"] = dataset_stacked["ze"].mean(dim="rho")

    # Calculate profile statistics.
    dataset_stacked["ne_avg"] = dataset_stacked["ne"].mean(dim="rho")
    dataset_stacked["ni_avg"] = dataset_stacked["ni"].mean(dim="rho")
    dataset_stacked["ne19"] = 1e-19 * dataset_stacked["ne_avg"]
    dataset_stacked["ni19"] = 1e-19 * dataset_stacked["ni_avg"]
    dataset_stacked["ne19_prof"] = 1e-19 * dataset_stacked["ne"]
    dataset_stacked["ni19_prof"] = 1e-19 * dataset_stacked["ni"]

    # For certain variables, such as Ip, RAPTOR stores values across the grid, for the amount of
    # the quantity enclosed in each flux surface, but we only care about the total value.
    dataset_stacked["Ip"] = dataset_stacked["Ip"].isel(rho=-1)
    dataset_stacked["Ioh"] = dataset_stacked["Ioh"].isel(rho=-1)
    dataset_stacked["epsilon"] = dataset_stacked["epsilon"].isel(rho=-1)
    dataset_stacked["aminor"] = dataset_stacked["epsilon"] * shot_constants.R0
    dataset_stacked["Ip_MA"] = dataset_stacked["Ip"] * 1e-6
    dataset_stacked["li"] = 2.0 * dataset_stacked["Li"] / (constants.MU0 * shot_constants.R0)

    # NOTE: RAPTOR has COCOS=12, so the sign of psi is flipped from what we expect.
    dataset_stacked["psi"] = -1.0 * dataset_stacked["psi"]
    dataset_stacked["psib"] = dataset_stacked["psi"].isel(rho=-1)

    # kappa from RAPTOR is a profile, rename it to kappa_profile and make "kappa" refer to the edge.
    dataset_stacked["kappa_profile"] = dataset_stacked["kappa"]
    dataset_stacked["kappa"] = dataset_stacked["kappa"].isel(rho=-1)
    dataset_stacked["surface_area"] = dataset_stacked["Volume"].isel(rho=-1) / (
        2.0 * np.pi * shot_constants.R0
    )
    dataset_stacked["kappa_a"] = dataset_stacked["surface_area"] / (
        np.pi * dataset_stacked["aminor"]**2.0
    )

    # For some reason we don't need to flip signs on the loop voltage.
    dataset_stacked["upl"] = dataset_stacked["upl"]
    dataset_stacked["vb"] = dataset_stacked["upl"].isel(rho=-1)

    # Euqation 9 from Romero.
    dataset_stacked["psic"] = (
        2 * dataset_stacked["Wpol"] + dataset_stacked["psib"] * dataset_stacked["Ip"]
    ) / dataset_stacked["Ip"]

    dataset = dataset_stacked.unstack(dim="time_slices")

    dataset["Vind"] = apply_episodewise_function(dataset, lambda eps: vind_24(eps))
    dataset["vc"] = apply_episodewise_function(
        dataset, lambda eps: -1.0 * eps["psic"].differentiate("time")
    )
    dataset["Ip_dot"] = apply_episodewise_function(
        dataset, lambda eps: eps["Ip"].differentiate("time")
    )
    dataset["Li_dot"] = apply_episodewise_function(
        dataset, lambda eps: eps["Li"].differentiate("time")
    )
    dataset["dWdt_ff"] = apply_episodewise_function(
        dataset, lambda eps: eps["Wth"].differentiate("time")
    )

    dataset["vc_minus_vb"] = dataset["vc"] - dataset["vb"]
    dataset["vc_minus_vb_2"] = (
        -dataset["Li_dot"] * dataset["Ip"] - dataset["Ip_dot"] * dataset["Li"]
    )
    dataset["vc2"] = dataset["vc_minus_vb_2"] + dataset["vb"]
    return dataset


def load_data():
    """Load the data."""
    # Load the data.
    path = os.path.join(pop_down_gym.ROOT_DIR, "data/raptor_sim_for_rl.nc")
    ds = preprocess(xr.load_dataset(path), 31)
    return ds