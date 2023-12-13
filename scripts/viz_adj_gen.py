import copy
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import numpy as np
import orbax
import orbax.checkpoint
import tqdm
import typer
from loguru import logger

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

    if (ckpt_dir / "checkpoint").exists() and (ckpt_dir / "checkpoint").is_file():
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ppo_dict = orbax_checkpointer.restore(ckpt_dir, item={"ppo": ppo})
        ppo: PPOAlg = ppo_dict["ppo"]
    else:
        ckpt_manager = get_ckpt_manager_sync(ckpt_dir, max_to_keep=100)
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

    ########################################################
    num_per_range = 60
    interp_fracs = np.linspace(-1.0, 1.0, num=num_per_range)
    interp_fracs0 = np.linspace(0.0, 1.0, num=num_per_range)

    zero_offset_dict = {key: 0.0 for key in shift_ranges.keys()}
    offset_series = []

    # First, for each constraint, go from -1 to 1.
    for key in shift_ranges.keys():
        b_offsets = []
        for interp_frac in interp_fracs:
            offset_dict = zero_offset_dict.copy()
            offset_dict[key] = interp_frac * shift_ranges[key]
            b_offsets.append(offset_dict)
        offset_series.append(b_offsets)

    # Next, sample two random points.
    rng = np.random.default_rng(seed=58124)
    pt0 = np.full(len(shift_ranges), -1.0)
    pt1 = rng.uniform(-1.0, 1.0, size=len(shift_ranges))
    pt2 = rng.uniform(-1.0, 1.0, size=len(shift_ranges))

    def interp_pt(p1, p2, frac):
        # frac: in [0, 1].
        offset_dict = zero_offset_dict.copy()
        for ii, (key, shift) in enumerate(shift_ranges.items()):
            offset_dict[key] = ((1 - frac) * p1[ii] + frac * p2[ii]) * shift
        return offset_dict

    # Interpolate from 0 -> pt1
    b_offsets = []
    for interp_frac in interp_fracs0:
        b_offsets.append(interp_pt(pt0, pt1, interp_frac))
    offset_series.append(b_offsets)

    # pt1 -> pt2
    b_offsets = []
    for interp_frac in interp_fracs0:
        b_offsets.append(interp_pt(pt1, pt2, interp_frac))
    offset_series.append(b_offsets)

    # pt2 -> 0
    b_offsets = []
    for interp_frac in interp_fracs0:
        b_offsets.append(interp_pt(pt2, pt0, interp_frac))
    offset_series.append(b_offsets)

    pkl_path = plot_dir / "fancy_data_offsets.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(offset_series, f)
    logger.info("Saved to {}!".format(pkl_path))

    ##################################################################
    # Run!
    eval_series = []
    for b_offsets in tqdm.tqdm(offset_series):
        b_eval = []
        for offset_dict in b_offsets:
            b_eval.append(jax2np(test_for_param(offset_dict)))
        eval_series.append(b_eval)

    # Save the data.
    pkl_path = plot_dir / "fancy_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(eval_series, f)

    logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
