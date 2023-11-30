import copy
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import typer
from loguru import logger
from matplotlib.collections import LineCollection

from jaxrl.ppo import Collector, CollectorCfg, PPOAlg, PPOCfg
from jaxrl.utils.ckpt_manager import get_ckpt_manager_sync
from jaxrl.utils.jax_utils import jax2np
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj


def main(ckpt_dir: pathlib.Path):
    rew_bounds = {
        "li": [2, 3],
        "ng_frac": [0.5, 0.8],
        "beta_n": [0.015, 0.028],
        "beta_p": [0.25, 0.4],
        "Bv_dot_mag": [0.2, 0.4],
        "Wdot_mag": [20_000_000, 70_000_000],
        "shafranov_coeff": [3.4, 3.6],
        "iota95": [0.35, 0.45],
    }
    rew_centers = {k: 0.5 * (v[0] + v[1]) for k, v in rew_bounds.items()}
    shift_ranges = {k: 0.5 * (v[1] - v[0]) for k, v in rew_bounds.items()}

    ckpt_manager = get_ckpt_manager_sync(ckpt_dir, max_to_keep=100)

    ppo_cfg = PPOCfg(
        pol_lr=3e-4,
        val_lr=3e-4,
        entropy_cf=1.0,
        disc_gamma=0.99,
        pol_hid_sizes=[256, 256, 256],
        val_hid_sizes=[256, 256, 256],
        act="tanh",
        pol_type="TanhNormal",
        train_cfg=None,
        rew_scale=5e2,
        clip_grad=1.0,
    )
    env = PDEnvAdj(shift_ranges=shift_ranges, limits=rew_centers, shift_mult=0)
    ppo = PPOAlg.create(jr.PRNGKey(5123), env, ppo_cfg)

    step = ckpt_manager.latest_step()
    ppo_dict = ckpt_manager.restore(step, items={"ppo": ppo})
    ppo: PPOAlg = ppo_dict["ppo"]

    plot_dir = ckpt_dir.parent / "eval_plots"
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
