import copy
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import numpy as np
import typer
from loguru import logger

from jaxrl.helpers import get_constr_vals_from_interp, get_default_rew_bounds, load_ppo
from jaxrl.ppo import Collector, CollectorCfg
from jaxrl.utils.jax_utils import jax2np


def main(ckpt_dir: pathlib.Path):
    rew_centers, shift_ranges, rew_min, rew_max = get_default_rew_bounds()
    n_constr = len(rew_centers)

    env, ppo, plot_dir = load_ppo(ckpt_dir, shift_ranges, rew_centers)
    plot_dir.mkdir(exist_ok=True, parents=True)

    @jax.jit
    def test_for_param(offset_dict_: dict):
        env_test = copy.copy(env)
        env_test.shift_ranges = shift_ranges
        env_test.offset = offset_dict_
        env_test.shift_mult = 0.0

        collect_cfg = CollectorCfg(0, 0, n_env_eval=128, rollout_T_eval=120)
        collector = Collector.create(jr.PRNGKey(1234), env_test, collect_cfg)
        return ppo.eval(collector)

    # Sample random params. First one is more loose, second one is more tight.
    rng = np.random.default_rng(seed=592130)
    interps = rng.uniform(-0.9, 0.9, size=(3, n_constr))
    interp_lo = np.min(interps, axis=0)
    interp_hi = np.max(interps, axis=0)
    interp_lo[0] = -0.85
    print("hi: {}".format(interp_hi))
    print("lo: {}".format(interp_lo))

    offset_dict_lo, val_dict_lo = get_constr_vals_from_interp(interp_lo, shift_ranges, rew_centers)
    offset_dict_hi, val_dict_hi = get_constr_vals_from_interp(interp_hi, shift_ranges, rew_centers)

    logger.info("Testing for lo...")
    data_lo = jax2np(test_for_param(offset_dict_lo))
    logger.info("Testing for hi...")
    data_hi = jax2np(test_for_param(offset_dict_hi))
    logger.info("Testing for hi... done!")

    # Save the data.
    dict_lo = {"data": data_lo, "offset": offset_dict_lo, "val": val_dict_lo}
    dict_hi = {"data": data_hi, "offset": offset_dict_hi, "val": val_dict_hi}
    dict_all = [dict_lo, dict_hi]

    pkl_path = plot_dir / "data_plot.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dict_all, f)
    logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
