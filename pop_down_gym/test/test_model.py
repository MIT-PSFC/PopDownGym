import jax
import jax.numpy as jnp
import numpy as np
from pop_down_gym.model import Model
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
        "dgs_dt": 0.2 * jnp.ones(nt),
    }

    params = {
        "Hmode": True,
        "Hfactor": 1.0,
        "Zeff": 1.5,
        "ion_dilution": 0.85,
        "Te_over_Ti": 1.2,
        "f_dt": 0.5,
        "tau_n_factor": 7.5,
        "prad_mult": 2.0,
    }
    test, ds = Model.create_default()

    test(initial_state, jax.tree_map(lambda x: x[0], controls), params=params)

    sim = SimFFControl(test, dt0=1e-2)
    res = sim.simulate(ts, initial_state, controls, params=params)


def test_simulate():
    # debug nans
    jax.config.update("jax_debug_nans", True)
    # disable jit.
    # jax.config.update("jax_disable_jit", True)
    model, ds = Model.create_default()
    ts = ds["time"].to_numpy()
    ds0 = ds.isel(time=0)

    initial_state = {
        "li": ds0["li"].to_numpy().squeeze(),
        "Ip_MA": ds0["Ip_MA"].to_numpy().squeeze(),
        "vc_minus_vb": ds0["vc_minus_vb"].to_numpy().squeeze(),
        "Wth": ds0["Wth"].to_numpy().squeeze(),
        "nfuel19_vol": ds0["ni19_vol_avg"].to_numpy().squeeze(),
        "Paux": ds0["Paux_MW"].to_numpy().squeeze(),
        "gs": 0.0,
    }

    controls = {
        "dIp_dt": ds["Ip_MA"].differentiate("time").to_numpy().squeeze(),
        "dPaux_dt": ds["Paux_MW"].differentiate("time").to_numpy().squeeze(),
        "fueling19": np.zeros(ds["time"].shape),  # TODO(allenw): not self consistent.
        "dgs_dt": 0.2 * np.ones(ds["time"].shape),
    }
    Hmode = ds["Hmode"].to_numpy().squeeze()

    params = {}

    sim = SimFFControl(model, dt0=1e-2)

    params = {
        "Hmode": True,
        "Hfactor": 1.0,
        "Zeff": 1.5,
        "ion_dilution": 0.85,
        "Te_over_Ti": 1.2,
        "f_dt": 0.5,
        "tau_n_factor": 7.5,
        "prad_mult": 2.0,
    }

    @jax.jit
    def simulate(ts_step, state, controls, params):
        return sim.simulate(ts_step, state, controls, params=params)

    states = [initial_state]
    for i in range(ts.size - 1):
        ts_step = ts[i : i + 2]
        controls_step = jax.tree_map(lambda x: x[i : i + 2], controls)
        params["Hmode"] = Hmode[i]
        res = simulate(ts_step, states[-1], controls_step, params=params)
        new_state = jax.tree_map(lambda x: x[-1], res.ys)
        states.append(new_state)
        print(i)

    def tree_transpose(list_of_trees):
        """Convert a list of trees of identical structure into a single tree of lists."""
        return jax.tree_map(lambda *xs: jnp.array(xs), *list_of_trees)

    states = tree_transpose(states)

    n_vars = len(list(states.keys()))
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(n_vars, 1, figsize=(15, n_vars * 10))
    for i, (var, state) in enumerate(states.items()):
        axs[i].plot(ts, state)
        axs[i].set_title(var)
    plt.show()
    import pdb

    pdb.set_trace()
