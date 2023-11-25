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
