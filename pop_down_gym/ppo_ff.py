import functools as ft
from typing import Any, Literal, NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from attrs import define
from flax import struct
from loguru import logger

from jaxrl.env import Env, StepOutput
from jaxrl.env_types import BObs, BState, BTControl, BTState, Control, Obs, TControl, TObs
from jaxrl.networks.mlp import MLP
from jaxrl.networks.network_utils import ActLiteral, HidSizes, get_act_from_str
from jaxrl.networks.policy import NormalPolicyStdConst, TanhNormalPolicy, TanhTransformedDistribution
from jaxrl.networks.value import ValueNet
from jaxrl.ppo import MovingAverage, PPOAlg, PPOCfg
from jaxrl.utils.grad_utils import compute_norm, compute_norm_and_clip
from jaxrl.utils.jax_types import BTBool, BTFloat, FloatScalar, IntScalar, MetricsDict, PRNGKey, TBool, TFloat
from jaxrl.utils.jax_utils import concat_at_front, jax_vmap, merge01, tree_split_dims
from jaxrl.utils.schedule import Schedule, as_schedule
from jaxrl.utils.tfp import tfd
from jaxrl.utils.train_state import TrainState
from pop_down_gym.tv_encoder import TVEncoder


def make_ppo_ff(key: PRNGKey, env: Env, cfg: PPOCfg):
    logger.info("Making PPO FF...")

    key, key_pol, key_V = jr.split(key, 3)

    act = get_act_from_str(cfg.act)

    obs, obs_priv, state = env.reset(key)

    tvencoder_cfg = dict(time_min=env.dt, time_max=7.0, n_freqs=64)

    # Define policy network.
    pol_base_cls = ft.partial(MLP, cfg.pol_hid_sizes, act, act_final=True)
    enc_cls = ft.partial(MLP, [128], act, act_final=True)
    # Add time-dependent part.
    pol_base_cls = ft.partial(TVEncoder, enc_cls, pol_base_cls, **tvencoder_cfg)
    # Policy head.
    if cfg.pol_type == "Normal":
        pol_def = NormalPolicyStdConst(pol_base_cls, env.n_actions)
    elif cfg.pol_type == "TanhNormal":
        pol_def = TanhNormalPolicy(pol_base_cls, env.n_actions)
    else:
        raise NotImplementedError(f"{cfg.pol_type} is not implemented.")

    pol_tx = optax.inject_hyperparams(ft.partial(_optim, clip_grad=cfg.clip_grad))(cfg.pol_lr)
    pol = TrainState.create_from_def(key_pol, pol_def, (obs,), pol_tx)

    # Define V network.
    V_base_cls = ft.partial(MLP, cfg.val_hid_sizes, act, act_final=True)
    # Add time-dependent part.
    V_base_cls = ft.partial(TVEncoder, enc_cls, V_base_cls, **tvencoder_cfg)
    # Value head.
    V_def = ValueNet(V_base_cls)
    V_tx = _optim(cfg.val_lr, cfg.clip_grad)
    V = TrainState.create_from_def(key_V, V_def, (obs_priv,), V_tx)

    V_ma = MovingAverage.create()
    ent_cf_sched = as_schedule(cfg.entropy_cf).make()
    return PPOAlg(0, key, pol, V, V_ma, cfg.pol_lr, ent_cf_sched, env, cfg)


def _optim(learning_rate: float, clip_grad: float):
    eps = 1e-5
    return optax.chain(optax.clip_by_global_norm(clip_grad), optax.adam(learning_rate=learning_rate, eps=eps))
