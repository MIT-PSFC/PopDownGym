import pathlib
from typing import Callable, Optional

import attrs
import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.collections import LineCollection
from orbax.checkpoint.checkpoint_manager import DEFAULT_ITEM_NAME

import wandb
from jaxrl.env import Env
from jaxrl.ppo import Collector, CollectorCfg, PPOAlg, PPOCfg, PPOEval
from jaxrl.utils.ckpt_manager import get_checkpointer, get_ckpt_manager_sync
from jaxrl.utils.jax_types import PRNGKey
from jaxrl.utils.jax_utils import jax2np
from pop_down_gym.pd_gym_stateless import PopDownGymStateless
from pop_down_gym.reward import RewardModel


def reorder_wandb_name(wandb_name: str = None, num_width: int = 4, max_word_len: int = 5) -> str:
    name_orig = wandb.run.name
    assert name_orig is not None
    name_parts = name_orig.split("-")
    assert len(name_parts) == 3
    word0, word1, num = name_parts
    # If words are too long, then truncate them.
    word0, word1 = word0[:max_word_len], word1[:max_word_len]
    num = num.zfill(num_width)
    if wandb_name is not None:
        new_name = "{}-{}".format(num, wandb_name)
    else:
        new_name = "{}-{}-{}".format(num, word0, word1)
    wandb.run.name = new_name
    return new_name


def train_ppo(
    wandb_name: str,
    key: PRNGKey,
    env_train: Env,
    env_test: Env,
    ppo_cfg: PPOCfg,
    collect_cfg: CollectorCfg,
    project_name: str,
    warmstart: Optional[pathlib.Path] = None,
    make_ppo: Callable = None,
    n_iters:int=10_000
):

    key_ppo, key_collect = jr.split(key, 2)

    if make_ppo is None:
        make_ppo = PPOAlg.create

    ppo = make_ppo(key_ppo, env_train, ppo_cfg)

    if warmstart is not None:
        assert warmstart.exists()
        orbax_checkpointer = get_checkpointer()
        d = orbax_checkpointer.restore(
            warmstart / DEFAULT_ITEM_NAME, item={"ppo": ppo, "collector": None, "ppo_cfg": None, "collect_cfg": None}
        )
        ppo = d["ppo"]
        # Restart all steps to 0.
        logger.info(f"Warm starting from {warmstart}! Starting from step=0...")
        policy = ppo.policy.replace(step=0)
        V = ppo.V.replace(step=0)
        pol_lr_init = ppo_cfg.pol_lr
        ppo = ppo.replace(update_idx=0, policy=policy, V=V, pol_lr=pol_lr_init)

    collector_train = Collector.create(key_collect, env_train, collect_cfg)
    collector_test = Collector.create(key_collect, env_test, collect_cfg)

    log_every = 5
    eval_every = 50
    ckpt_every = 100

    cfg_total = {"ppo": attrs.asdict(ppo_cfg), "collect": attrs.asdict(collect_cfg)}
    wandb.init(project=project_name, config=cfg_total)
    reorder_wandb_name(wandb_name)

    run_dir = pathlib.Path(__file__).parent.parent / "runs/{}".format(wandb.run.name)
    ckpt_dir = run_dir / "ckpts"
    plot_dir = run_dir / "plots"

    ckpt_manager = get_ckpt_manager_sync(ckpt_dir, max_to_keep=100)

    for idx in range(n_iters):
        ppo: PPOAlg
        logger.info("collect...")
        collector_train, data = ppo.collect(collector_train)
        logger.info("update...")
        ppo, loss_info = ppo.update(data)
        logger.info("update... done!")

        if idx % log_every == 0:
            # Estimate the total variation between the new policy and the behavior policy.
            est_tv = ppo.estimate_tv(data)
            loss_info["tv"] = est_tv

            log_dict = {f"train/{k}": v for k, v in loss_info.items()}
            logger.info("[{:5}] - V: {:8.2e}, pol: {:8.2e}".format(idx, loss_info["loss_V"], loss_info["loss_pol"]))
            wandb.log(log_dict, step=idx)

        if idx % eval_every == 0:
            logger.info("Eval....")
            eval_data: PPOEval = jax2np(ppo.eval(collector_test))
            logger.info("Eval.... Done!")

            ##########################################
            # Plot
            plot(idx, env_train.pd.reward_model, eval_data, plot_dir)
            logger.info("Plot... Done!")
            ##########################################

            # Compute stats.
            bT_valid = eval_data.bT_valid_mask
            b_reward = np.where(bT_valid, eval_data.bT_rew, 0.0).sum(axis=1)
            rew_min, rew_mean, rew_max = b_reward.min(), b_reward.mean(), b_reward.max()

            bT_info = eval_data.bT_info
            b_hitgoal = np.any(np.where(bT_valid, bT_info["hit_goal"], 0), axis=1)
            p_goal = np.mean(b_hitgoal)

            b_oob = np.any(np.where(bT_valid, bT_info["out_of_bounds"], 0), axis=1)
            p_oob = np.mean(b_oob)

            bT_std = eval_data.bT_misc["std"].mean(-1)
            b_std_valid = bT_std.flatten()[eval_data.bT_valid_mask.flatten()]
            std_mean = b_std_valid.mean()

            # Compute probability of constraint violations.
            constr_ub = env_train.pd.reward_model.limits
            p_vio = {}
            for k, ub in constr_ub.items():
                bT_r = eval_data.bT_info["reward_inputs"][k]
                b_vio = np.any(np.where(bT_valid, bT_r >= ub, False), axis=1)
                p_vio[f"Probs/Constr/{k}"] = np.mean(b_vio)

            log_dict = {
                "TotalReward/Min": rew_min,
                "TotalReward/Mean": rew_mean,
                "TotalReward/Max": rew_max,
                "Probs/p_oob": p_oob,
                "Probs/p_goal": p_goal,
                "StdMean": std_mean,
                **p_vio,
            }
            log_dict = log_dict | eval_data.ppo_info
            log_dict = {f"eval/{k}": v.mean() for k, v in log_dict.items()}
            wandb.log(log_dict, step=idx)

        if idx % ckpt_every == 0:
            ckpt_manager.save_ez(
                idx, {"ppo": ppo, "collector": collector_train, "ppo_cfg": ppo_cfg, "collect_cfg": collect_cfg}
            )


def plot(idx: int, rew_model: RewardModel, data: PPOEval, plot_dir: pathlib.Path):
    dt = 0.05
    plot_dir.mkdir(exist_ok=True, parents=True)
    plot_path = plot_dir / f"plot_{idx:06d}.jpg"

    constr_labels = PopDownGymStateless.constr_labels()
    nconstr = len(constr_labels)
    constr_ub = rew_model.limits
    Ip_MA_tgt = rew_model.ip_ma["target"]

    bT_rew_inputs = data.bT_info["reward_inputs"]

    figsize = np.array([6, 1.2 * nconstr])
    fig, axes = plt.subplots(nconstr, layout="constrained", figsize=figsize, sharex=True, dpi=250)
    for ii, ax in enumerate(axes):
        label = constr_labels[ii]

        b_segs = []
        bT_rew_input = bT_rew_inputs[label]
        for bb, T_r in enumerate(bT_rew_input):
            # Truncate to valid only.
            if not np.all(data.bT_valid_mask[bb]):
                idx_first_invalid = data.bT_valid_mask[bb].argmin()
                T_r = T_r[:idx_first_invalid]

            T = len(T_r)
            T_ts = dt * np.arange(T)
            # (T, 2)
            segs = np.stack([T_ts, T_r], axis=1)
            b_segs.append(segs)

        col = LineCollection(b_segs, color="C1", lw=0.5, alpha=0.4)
        ax.add_collection(col)
        ax.autoscale_view()
        ax.set_ylabel(label, rotation=0, ha="right")

        # Plot the limits.
        ymin, ymax = ax.get_ylim()
        if label in constr_ub:
            # Expand the ymax a bit.
            yrange = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.1 * yrange)
            ymin, ymax = ax.get_ylim()

            if ymax > constr_ub[label]:
                ax.axhspan(constr_ub[label], ymax, color="C0", alpha=0.2)

        if label == "Ip_MA":
            ax.axhspan(ymin, Ip_MA_tgt, color="C5", alpha=0.2)

    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
