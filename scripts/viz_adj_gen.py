import copy
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import orbax
import tqdm
import typer
from loguru import logger
from matplotlib.collections import LineCollection

from jaxrl.helpers import load_ppo, get_default_rew_bounds
from jaxrl.ppo import Collector, CollectorCfg, PPOAlg, PPOCfg
from jaxrl.utils.ckpt_manager import get_ckpt_manager_sync
from jaxrl.utils.jax_utils import jax2np
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj


def main(ckpt_dir: pathlib.Path):
    rew_centers, shift_ranges, rew_min, rew_max = get_default_rew_bounds()

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

    interp_fracs = np.linspace(-1.0, 1.0, num=30)
    eval_datas = []
    for interp_frac in tqdm.tqdm(interp_fracs):
        # key = "beta_p"
        key = "li"
        offset = interp_frac * shift_ranges[key]
        offset_dict = {key: offset}
        data = jax2np(test_for_param(offset_dict))
        eval_datas.append(data)

    # Save the data.
    pkl_path = plot_dir / "{:05}_data_{}.pkl".format(step, key)
    with open(pkl_path, "wb") as f:
        pickle.dump((eval_datas, interp_fracs), f)

    logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
