import jax
import pytest

from pop_down_gym.geometry import Geometry
from pop_down_gym.load_data import load_data


@pytest.mark.skip(reason="test is currently broken TODO@allen-adastra")
def test_geom():
    ds = load_data()
    time = ds.time.values.squeeze()
    aminor = ds.aminor.values.squeeze()
    kappa_a = ds.kappa_a.values.squeeze()
    Vp = ds.Vp.values.squeeze()
    g = Geometry(1.85, time, aminor, kappa_a, Vp)
    gdotfun = jax.jacfwd(g)
    gdot = gdotfun(0.0)

    # Could we have anything more informative here?
    assert gdot is not None


if __name__ == "__main__":
    test_geom()
