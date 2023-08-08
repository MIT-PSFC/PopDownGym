import jax
from pop_down_gym.geometry import Geometry
from pop_down_gym.data.load import load_data
    
def test_geom():
    ds = load_data()
    time = ds.time.values.squeeze()
    aminor = ds.aminor.values.squeeze()
    kappa_a = ds.kappa_a.values.squeeze()
    Vp = ds.Vp.values.squeeze()
    g = Geometry(1.85, time, aminor, kappa_a, Vp)
    gdotfun = jax.jacfwd(g)
    gdot = gdotfun(0.0)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_geom()