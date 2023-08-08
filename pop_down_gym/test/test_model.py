import jax
import jax.numpy as jnp
import pop_down_gym.model as model
from contrax.simulate import SimFFControl

def test_model():
    # State: [li, Ip_MA, vc_minus_vb, Wth, nfuel19_vol, Paux, gs]
    # Control: [dIp_dt, dPaux_dt, fueling19, dgs_dt]
    # Params: [HMode, Hfactor, Zeff, ion_dilution, Te_over_Ti]
    initial_state = {
        "li": 0.757764,
        "Ip_MA": 8.7,
        "vc_minus_vb": 0.153183,
        "Wth": 2.482841e07,
        "nfuel19_vol": 27.0,
        "Paux": 14.0,
        "gs": 0.0,
    }

    nt = 10
    ts = jnp.linspace(0.0, 1.0, nt)

    controls = {
        "dIp_dt": -1.0 * jnp.ones(nt),
        "dPaux_dt": 0.0 * jnp.ones(nt),
        "fueling19": 0.0 * jnp.ones(nt),
        "dgs_dt": 0.2 * jnp.ones(nt)
    }

    params = {
        "Hmode": True,
        "Hfactor": 1.0,
        "Zeff": 1.5,
        "ion_dilution": 0.85,
        "Te_over_Ti": 1.2,
        "f_dt": 0.5,
        "tau_n_factor": 7.5,
        "prad_mult": 2.0
    }
    test, ds = model.Model.create_default()

    test(initial_state, jax.tree_map(lambda x: x[0], controls), params=params)

    sim = SimFFControl(test, dt0=1e-2)
    res = sim.simulate(ts, initial_state, controls, params=params)
    import pdb; pdb.set_trace()