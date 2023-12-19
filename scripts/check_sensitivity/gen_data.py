import copy
import pathlib
import pickle

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import orbax
import orbax.checkpoint
import tqdm
from loguru import logger

from jaxrl.ppo import Collector, CollectorCfg, PPOAlg, PPOCfg, PPOEval
from jaxrl.utils.jax_utils import jax2np
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj


def load_ppo(env):
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
    ppo = PPOAlg.create(jr.PRNGKey(0), env, ppo_cfg)

    root_dir = pathlib.Path(__file__).parent.parent.parent
    tmp_dir = root_dir / "tmp"
    ckpt_path = tmp_dir / "ppo_adj_ckpt"
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ppo_dict = orbax_checkpointer.restore(ckpt_path, item={"ppo": ppo})
    ppo: PPOAlg = ppo_dict["ppo"]
    return ppo


def main():
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

    ######################################################
    env = PDEnvAdj(shift_ranges=shift_ranges, limits=rew_centers, shift_mult=0)
    ppo = load_ppo(env)
    ######################################################
    n_vals = 32
    interp_fracs = np.linspace(-1.0, 1.0, num=n_vals)

    @jax.jit
    def test_for_param(offset_dict_: dict):
        # Create the env_test
        env_test = copy.copy(env)
        env_test.shift_ranges = shift_ranges
        env_test.offset = offset_dict_
        env_test.shift_mult = 0.0

        collect_cfg = CollectorCfg(0, 0, n_env_eval=1024, rollout_T_eval=120)
        collector = Collector.create(jr.PRNGKey(1234), env_test, collect_cfg)
        eval_data: PPOEval = ppo.eval(collector)

        # Get whether it hit the goal, and how long it took to reach the goal.
        bT_hit_goal = eval_data.bT_info["hit_goal"]
        b_has_hit_goal_ = jnp.any(bT_hit_goal, axis=1)
        b_hit_goal_steps_ = jnp.argmax(bT_hit_goal, axis=1)
        return b_has_hit_goal_, b_hit_goal_steps_

    offset_dict_min = {k: -v for k, v in shift_ranges.items()}
    offset_dict_mid = {k: 0.0 for k, v in shift_ranges.items()}
    offset_dict_max = {k: v for k, v in shift_ranges.items()}

    for offset_dict, offset_name in [(offset_dict_min, "min"), (offset_dict_mid, "mid"), (offset_dict_max, "max")]:
        data_dict = {}
        for rew_name in rew_bounds.keys():
            logger.info(f"Checking sens for {rew_name}...")
            data_dict_rew = {}
            for ii, interp_frac in enumerate(interp_fracs):
                logger.info(f"    Checking {ii + 1:3} / {n_vals:3}...")
                offset = interp_frac * shift_ranges[rew_name]

                offset_dict_ = offset_dict.copy()
                offset_dict_[rew_name] = offset

                b_has_hit_goal, b_hit_goal_steps = jax2np(test_for_param(offset_dict_))

                constr_val = rew_centers[rew_name] + offset
                data_dict_rew[constr_val] = (b_has_hit_goal, b_hit_goal_steps)
            data_dict[rew_name] = data_dict_rew

        # Save theval data.
        pkl_path = pathlib.Path(__file__).parent / "ppo_sens_data_{}.pkl".format(offset_name)
        with open(pkl_path, "wb") as f:
            pickle.dump(data_dict, f)
        logger.info("Saved pkl to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
