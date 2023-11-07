import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tqdm import tqdm

from contrax.simulate import SimFFControl
from pop_down_gym.model import Model


@pytest.mark.parametrize("variant", [lambda fn: fn, jax.jit])
def test_model_call(variant):
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
    test_model, _ = Model.create_default()

    state_derivatives = variant(test_model)(
        initial_state, jax.tree_map(lambda x: x[0], controls), params=params
    )

    # State derivatives should have the right shape
    assert jax.tree_util.tree_structure(
        state_derivatives
    ) == jax.tree_util.tree_structure(initial_state)


# TODO this test is currently broken
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

    # @jax.jit
    def simulate(ts_step, state, controls, params):
        return sim.simulate(ts_step, state, controls, params=params)

    states = [initial_state]
    for i in range(ts.size - 1):
        ts_step = ts[i : i + 2]
        controls_step = jax.tree_map(lambda x: x[i : i + 2], controls)
        params["Hmode"] = Hmode[i]
        import pdb; pdb.set_trace()
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


def benchmark_model_call(num_trials: int = 100):
    """Speed benchmarking for calculating state derivatives."""
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
    model, _ = Model.create_default()

    no_jit_times = []
    for _ in tqdm(range(num_trials)):
        start = time.perf_counter()
        model(initial_state, jax.tree_map(lambda x: x[0], controls), params=params)
        no_jit_times.append(time.perf_counter() - start)

    jit_times = []
    fn = jax.jit(model)
    # Burn-in
    fn(initial_state, jax.tree_map(lambda x: x[0], controls), params=params)
    # Run trials
    for _ in tqdm(range(num_trials)):
        start = time.perf_counter()
        fn(initial_state, jax.tree_map(lambda x: x[0], controls), params=params)
        jit_times.append(time.perf_counter() - start)

    print(f"Benchmarking Model.__call__ with {num_trials} trials")
    print(f"Without jit: {np.mean(no_jit_times)} +/- {np.std(no_jit_times)}")
    print(f"With jit: {np.mean(jit_times)} +/- {np.std(jit_times)}")

if __name__ == "__main__":
    test_simulate()
