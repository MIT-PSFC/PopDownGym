import copy
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import numpy as np
import typer
from loguru import logger

from jaxrl.helpers import get_default_rew_bounds, load_ppo
from jaxrl.ppo import Collector, CollectorCfg, PPOEval
from jaxrl.utils.jax_utils import jax2np


def main(ckpt_dir: pathlib.Path, name: str):
    rew_centers, shift_ranges, rew_min, rew_max = get_default_rew_bounds()
    n_constr = len(rew_centers)
    pkl_path = pathlib.Path(__file__).parent / f"ppo_sens_hlfactor_{name}.pkl"
    logger.info("Saving pkl to {}...".format(pkl_path))

    env, ppo, plot_dir = load_ppo(ckpt_dir, shift_ranges, rew_centers)
    env.shift_mult = 0.0
    env.offset = {"Wdot_mag": -shift_ranges["Wdot_mag"]}
    plot_dir.mkdir(exist_ok=True, parents=True)

    @jax.jit
    def test_for_envparam(envparam_dict_: dict[str, float]):
        # Create the env_test
        env_test = copy.copy(env)
        env_test.env_params = envparam_dict_

        collect_cfg = CollectorCfg(0, 0, n_env_eval=1024, rollout_T_eval=120)
        collector = Collector.create(jr.PRNGKey(1234), env_test, collect_cfg)
        eval_data: PPOEval = ppo.eval(collector)
        return eval_data

    envparam_ranges: dict[str, tuple[float, float]] = env.pd.random_param_ranges
    hl_factor_range = envparam_ranges["hl_factor"]

    eval_datas = {}
    n_vals = 32
    interp_fracs = np.linspace(0.0, 1.0, num=n_vals)
    for ii, interp_frac in enumerate(interp_fracs):
        logger.info(f"    Checking {ii + 1:3} / {n_vals:3}...")
        lo, hi = hl_factor_range
        envparam_val = (1 - interp_frac) * lo + interp_frac * hi

        envparam_dict = {"hl_factor": envparam_val}
        eval_data = jax2np(test_for_envparam(envparam_dict))
        eval_datas[envparam_val] = eval_data

    # Save the data.
    with open(pkl_path, "wb") as f:
        pickle.dump(eval_datas, f)
    logger.info("Saved pkl to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
