from pop_down_gym.load_data import load_data
from pop_down_gym.profiles import ProfileBases
import equinox as eqx


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
    
    # Test that the bases are split right.
    trainable, static = eqx.partition(hmode_basis, hmode_basis.trainable_params_filter_spec())

    # Check the Te_basis is split right.
    assert trainable.Te_basis.v is not None and trainable.Te_basis.b is not None
    assert trainable.Te_basis.rhogauss is None and trainable.Te_basis.wgauss is None
    assert static.Te_basis.v is None and static.Te_basis.b is None
    assert static.Te_basis.rhogauss is not None and static.Te_basis.wgauss is not None

    # Check the Ti_basis is split right.
    assert trainable.Ti_basis.v is not None and trainable.Ti_basis.b is not None
    assert trainable.Ti_basis.rhogauss is None and trainable.Ti_basis.wgauss is None
    assert static.Ti_basis.v is None and static.Ti_basis.b is None
    assert static.Ti_basis.rhogauss is not None and static.Ti_basis.wgauss is not None

    # Check the ne_basis is split right.
    assert trainable.ne_basis.v is not None and trainable.ne_basis.b is not None
    assert trainable.ne_basis.rhogauss is None and trainable.ne_basis.wgauss is None
    assert static.ne_basis.v is None and static.ne_basis.b is None
    assert static.ne_basis.rhogauss is not None and static.ne_basis.wgauss is not None

    # Check the ni_basis is split right.
    assert trainable.ni_basis.v is not None and trainable.ni_basis.b is not None
    assert trainable.ni_basis.rhogauss is None and trainable.ni_basis.wgauss is None
    assert static.ni_basis.v is None and static.ni_basis.b is None
    assert static.ni_basis.rhogauss is not None and static.ni_basis.wgauss is not None