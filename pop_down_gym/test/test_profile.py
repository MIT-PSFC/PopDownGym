from pop_down_gym.load_data import load_data
from pop_down_gym.profiles import ProfileBases


def test_simple_profile_basis():
    ds, _ = load_data()
    hmode = ds["te"].sel(rho=0.95, method="nearest") > 3000
    ds["Hmode"] = hmode.drop_vars("rho")  #
    hmode_data = ds.where(ds["Hmode"], drop=True)
    lmode_data = ds.where(~ds["Hmode"], drop=True)
    hmode_basis, _, _ = ProfileBases.from_dataset(hmode_data)
    lmode_basis, _, _ = ProfileBases.from_dataset(lmode_data)

    eps_ds = ds.isel(episode=0)
    for i in range(eps_ds["time"].to_numpy().size):
        time_slice = eps_ds.isel(time=i)
        if time_slice["Hmode"]:
            basis = hmode_basis
        else:
            basis = lmode_basis

        Vp = time_slice["Vp"].values.squeeze()

        # Test Te transforms.
        te_kev_vol_avg = basis.Te_basis.volume_average(
            time_slice["te_kev"].values.squeeze(), Vp
        )
        te_kev_profile = basis.Te_basis.volume_average_to_profile(
            time_slice["te_kev_vol_avg"].values.squeeze(),
            Vp,
        )

        # Could we have anything more informative here?
        assert te_kev_vol_avg is not None
        assert te_kev_profile is not None