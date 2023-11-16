import pytest
import xarray as xr

from pop_down_gym.load_data import load_data
from pop_down_gym.profiles import ProfileBases, SimpleProfileBasis


@pytest.mark.skip(reason="test is currently broken TODO@allen-adastra")
def test_simple_profile_basis():
    ds = load_data()
    hmode = ds["te"].sel(rho=0.95, method="nearest") > 3000
    ds["Hmode"] = hmode.drop_vars("rho")  #
    hmode_data = ds.where(ds["Hmode"], drop=True)
    lmode_data = ds.where(~ds["Hmode"], drop=True)
    hmode_basis, _, _ = ProfileBases.from_dataset(hmode_data)
    lmode_basis, _, _ = ProfileBases.from_dataset(lmode_data)

    for i in range(ds["time"].to_numpy().size):
        slice = ds.isel(time=i)
        if slice["Hmode"]:
            basis = hmode_basis
        else:
            basis = lmode_basis

        Vp = slice["Vp"].values.squeeze()

        # Test Te transforms.
        te_kev_vol_avg = basis.Te_basis.volume_average(
            slice["te_kev"].values.squeeze(), Vp
        )
        te_kev_profile = basis.Te_basis.volume_average_to_profile(
            slice["te_kev_vol_avg"].values.squeeze(),
            Vp,
        )

        # Could we have anything more informative here?
        assert te_kev_vol_avg is not None
        assert te_kev_profile is not None


if __name__ == "__main__":
    test_simple_profile_basis()
