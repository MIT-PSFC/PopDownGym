import jax
import jax.numpy as jnp

from pop_down_gym.geometry import Geometry
from pop_down_gym.load_data import load_data

def test_geom():
    ds, ds_geom = load_data()
    time = ds_geom.time.values.squeeze()
    aminor = ds_geom.aminor.values.squeeze()
    kappa_a = ds_geom.kappa_a.values.squeeze()
    Vp = ds_geom.Vp.values.squeeze()
    g = Geometry(1.85, time, aminor, kappa_a, Vp)
    gdotfun = jax.jacfwd(g)

    gval = g(0.0)
    gdotval = gdotfun(0.0)

    # Sanity check values of gval.
    assert gval["aminor"] > 0.5 and gval["aminor"] < 0.6
    assert gval["kappa_a"] > 1.5 and gval["kappa_a"] < 1.8
    assert gval["Vp"].size > 20
    assert gval["Vp"][0] > 0.0 and gval["Vp"][0] < 1.0 # dV/drho near core is small.
    assert gval["Vp"][-1] > 30.0 and gval["Vp"][-1] < 40.0 # dV/drho near edge is large.
    assert gval["volume"] > 19.0 and gval["volume"] < 21.0

    # Sanity check values of gdotval.
    assert gdotval["aminor"] > 0.0 and gdotval["aminor"] < 0.1 # Expect aminor to increase a bit temporarily.
    assert gdotval["kappa_a"] < 0.0
    assert gdotval["Vp"].size > 20
    assert jnp.all(gdotval["Vp"] < 0.0) # Expect Vp to decrease everywhere.
    assert gdotval["volume"] < 0.0 # Expect volume to decrease.