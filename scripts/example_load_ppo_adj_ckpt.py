import pathlib
import ipdb
import jax
import jax.lax as lax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import orbax
import orbax.checkpoint
from loguru import logger

from jaxrl.env import StepOutput
from jaxrl.ppo import PPOAlg, PPOCfg
from pop_down_gym.pd_gym_jaxrl import PDEnvAdj

def default_build_ppo():
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

    #####################################
    # Shift the constraint boundary during test time.
    perturb_dict = {"li": 2.0}
    offset_dict = {k: v - rew_centers[k] for k, v in perturb_dict.items()}
    for k, v in perturb_dict.items():
        logger.info("Testing with {} = {} (offset={})".format(k, v, offset_dict[k]))
    #####################################

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
    env = PDEnvAdj(shift_ranges=shift_ranges, offset=offset_dict, limits=rew_centers, shift_mult=0)
    ppo = PPOAlg.create(jr.PRNGKey(0), env, ppo_cfg)

    root_dir = pathlib.Path(__file__).parent.parent.parent
    tmp_dir = root_dir / "tmp"
    ckpt_path = tmp_dir / "ppo_adj_ckpt"
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ppo_dict = orbax_checkpointer.restore(ckpt_path, item={"ppo": ppo})
    ppo: PPOAlg = ppo_dict["ppo"]
    return ppo, env, offset_dict, tmp_dir

def main():
    ppo, env, offset_dict, tmp_dir = default_build_ppo()

    ######################################################
    # Example of using the loaded policy.
    obs, obs_priv, env_state = env.reset(jr.PRNGKey(1337))
    action = ppo.act(obs)
    print("Obs: {}".format(obs))
    print("Action: {}".format(action))

    ######################################################
    # Example of a rollout + plotting.
    def rollout(env_state__, obs__):
        def rollout_body(stateobs, _):
            env_state_, obs_ = stateobs
            action_ = ppo.act(obs_)
            res = env.step_env(jr.PRNGKey(0), env_state_, action_)
            return (res.state, res.obs), res

        stateobs_init = (env_state__, obs__)
        _, T_res_ = lax.scan(rollout_body, stateobs_init, None, length=100)
        return T_res_

    rollout_jit = jax.jit(rollout)
    T_res: StepOutput = rollout_jit(env_state, obs)
    T_rew_inputs = T_res.info["reward_inputs"]
    T_invalid = np.cumsum(T_res.terminated) > 0
    T_valid_mask = np.concatenate([np.ones(1, dtype=bool), ~T_invalid[:-1]], axis=0)
    assert T_valid_mask.shape == (len(T_invalid),)

    ############################################################
    # Plot.
    plot_dir = tmp_dir
    plot_path = plot_dir / "ppo_adj_example_traj.pdf"

    constr_ub = env.pd.reward_model.limits
    Ip_MA_tgt = env.pd.reward_model.ip_ma["target"]

    constr_labels = env.constr_labels
    nconstr = len(constr_labels)
    figsize = np.array([6, 1.2 * nconstr])
    fig, axes = plt.subplots(nconstr, layout="constrained", figsize=figsize, sharex=True, dpi=250)
    for ii, ax in enumerate(axes):
        label = constr_labels[ii]

        T_r = T_rew_inputs[label]

        if not np.all(T_valid_mask):
            idx_first_invalid = T_valid_mask.argmin()
            T_r = T_r[:idx_first_invalid]

        T_t = np.arange(len(T_r)) * env.pd.dt
        ax.plot(T_t, T_r, color="C1", alpha=0.95)
        ax.set_ylabel(label, rotation=0, ha="right")

        # Plot the limits.
        ymin, ymax = ax.get_ylim()
        if label in constr_ub:
            constr_ub_ = constr_ub[label]

            if label in offset_dict:
                constr_ub_ = constr_ub_ + offset_dict[label]

            # Expand the ymax a bit.
            yrange = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.1 * yrange)
            ymin, ymax = ax.get_ylim()

            if ymax > constr_ub_:
                ax.axhspan(constr_ub_, ymax, color="C0", alpha=0.2)

        if label == "Ip_MA":
            ax.axhspan(ymin, Ip_MA_tgt, color="C5", alpha=0.2)

    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
