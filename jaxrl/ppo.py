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

from jaxrl.env import Env, StepOutput
from jaxrl.env_types import BObs, BState, BTControl, BTState, Control, Obs, TControl, TObs
from jaxrl.networks.mlp import MLP
from jaxrl.networks.network_utils import ActLiteral, HidSizes, get_act_from_str
from jaxrl.networks.policy import NormalPolicyStdConst, TanhNormalPolicy, TanhTransformedDistribution
from jaxrl.networks.value import ValueNet
from jaxrl.utils.grad_utils import compute_norm, compute_norm_and_clip
from jaxrl.utils.jax_types import BTBool, BTFloat, FloatScalar, IntScalar, MetricsDict, PRNGKey, TBool, TFloat
from jaxrl.utils.jax_utils import concat_at_front, jax_vmap, merge01, tree_split_dims
from jaxrl.utils.schedule import Schedule, as_schedule
from jaxrl.utils.tfp import tfd
from jaxrl.utils.train_state import TrainState


@define
class CollectorCfg:
    # Batch size when collecting data.
    n_envs: int
    # How long to rollout before updating.
    rollout_T: int

    n_env_eval: int
    rollout_T_eval: int


@define
class PPOTrainCfg:
    gae_lambda: float

    # Batch size during training.
    batch_size: int
    # How many epochs to use per update.
    n_update_epochs: int

    # PPO clip ratio
    clip_ratio: float

    # Desired KL for adaptive lr.
    kl_desired: float
    # Max pol LR when adapting.
    pol_lr_max: float
    # Min pol LR  when adapting.
    pol_lr_min: float

    # If true, normalize targets so that output of V is centered.
    normalize_V: bool = False


@define
class PPOCfg:
    pol_lr: float
    val_lr: float
    # In units of update_idx.
    entropy_cf: Schedule | float
    disc_gamma: float

    pol_hid_sizes: HidSizes
    val_hid_sizes: HidSizes

    act: ActLiteral
    pol_type: Literal["Normal", "TanhNormal"]

    train_cfg: PPOTrainCfg

    # Scale all rewards coming from the environment.
    rew_scale: float = 1.0
    clip_grad: float = 1.0


def _optim(learning_rate: float, clip_grad: float):
    eps = 1e-5
    return optax.chain(optax.clip_by_global_norm(clip_grad), optax.adam(learning_rate=learning_rate, eps=eps))


class CollectorState(NamedTuple):
    b_obs: BObs
    b_obspriv: BObs
    b_state: BState


class RolloutOutput(NamedTuple):
    Tp1_obs: TObs
    Tp1_obs_priv: TObs
    T_action: TControl
    T_logprob: TFloat
    T_reward: TFloat
    T_terminated: TBool
    T_truncated: TBool
    T_info: Any


class Collector(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    collect_state: CollectorState
    env: Env = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)

    @classmethod
    def create(cls, key: PRNGKey, env: Env, cfg: CollectorCfg):
        key, key_init = jr.split(key)
        b_obs, b_obspriv, b_state = jax.vmap(env.reset)(jr.split(key_init, cfg.n_envs))
        return Collector(0, key, CollectorState(b_obs, b_obspriv, b_state), env, cfg)

    def collect_single(
        self, key0: PRNGKey, collectstate0: CollectorState, get_pol
    ) -> tuple[CollectorState, RolloutOutput]:
        def _body(state: CollectorState, key):
            pol: tfd.Distribution = get_pol(state.b_obs)
            action, logprob = pol.experimental_sample_and_log_prob(seed=key)
            new = self.env.step_autoreset(key, state.b_state, action)
            return CollectorState(new.obs, new.obs_priv, new.state), (new, action, logprob)

        T_keys = jr.split(key0, self.cfg.rollout_T)
        collect_state: CollectorState
        T_outputs: StepOutput
        collect_state, (T_outputs, T_u, T_logprob) = lax.scan(_body, collectstate0, T_keys, length=self.cfg.rollout_T)

        # Add the initial observations.
        Tp1_obs = concat_at_front(collect_state.b_obs, T_outputs.obs)
        Tp1_obspriv = concat_at_front(collect_state.b_obspriv, T_outputs.obs_priv)
        T_out = RolloutOutput(
            Tp1_obs,
            Tp1_obspriv,
            T_u,
            T_logprob,
            T_outputs.reward,
            T_outputs.terminated,
            T_outputs.truncated,
            T_outputs.info,
        )

        return collect_state, T_out

    def collect_batch(self, get_pol) -> tuple["Collector", RolloutOutput]:
        key0 = jr.fold_in(self.key, self.collect_idx)
        b_keys = jr.split(key0, self.cfg.n_envs)
        collect_fn = ft.partial(self.collect_single, get_pol=get_pol)
        collect_state, bT_outputs = jax.vmap(collect_fn)(b_keys, self.collect_state)
        new_self = self.replace(collect_idx=self.collect_idx + 1, collect_state=collect_state)
        return new_self, bT_outputs

    def rollout_eval_single(self, get_pol, key0: PRNGKey):
        def body(stateobs, key):
            state_, obs_ = stateobs
            control, misc = get_pol(obs_)
            step_out = self.env.step_autoreset(key, state_, control)
            return (step_out.state, step_out.obs), (state_, step_out, control, misc)

        key_init, key_other = jr.split(key0, 2)
        obs, obspriv, state = self.env.reset(key_init)
        T_keys = jr.split(key_other, self.cfg.rollout_T_eval)
        stateobs_init = (state, obs)
        _, (T_state, T_outs, T_control, T_misc) = lax.scan(body, stateobs_init, T_keys, length=self.cfg.rollout_T_eval)

        return T_state, T_outs, T_control, T_misc

    def rollout_eval(self, get_pol) -> tuple[BTState, StepOutput, BTControl, dict]:
        key0 = jr.PRNGKey(314159)
        b_keys = jr.split(key0, self.cfg.n_env_eval)
        bT_state, bT_outs, bT_control, bT_misc = jax.vmap(ft.partial(self.rollout_eval_single, get_pol))(b_keys)
        return bT_state, bT_outs, bT_control, bT_misc


class PPOBatch(NamedTuple):
    obs: Obs
    obs_priv: Obs
    control: Control
    logprob: FloatScalar
    advantage: FloatScalar
    q: FloatScalar


class PPOEval(NamedTuple):
    # True until the first time the rollout is terminated or truncated.
    bT_valid_mask: BTBool
    bT_state: BTState
    bT_control: BTControl
    bT_rew: BTFloat
    bT_info: dict
    bT_misc: dict
    ppo_info: dict


def compute_gae(gamma: float, lambd: float, T_rew: TFloat, Tp1_V: TFloat, T_term: TBool) -> tuple[TFloat, TFloat]:
    def body(gae, deltaterm):
        delta, term = deltaterm
        gae_prev = delta + gamma * lambd * gae * (1 - term)
        return gae_prev, gae_prev

    (T,) = T_rew.shape
    T_V_next = Tp1_V[1:]
    T_V_curr = Tp1_V[:-1]
    T_delta = T_rew + gamma * T_V_next * (1 - T_term) - T_V_curr

    deltaterm_input = T_delta[:-1], T_term[:-1]
    _, Tm1_gae = lax.scan(body, T_delta[-1], deltaterm_input, length=T - 1, reverse=True)

    T_A_gae = jnp.concatenate([Tm1_gae, T_delta[-1, None]], axis=0)
    assert T_A_gae.shape == (T,)
    T_Q_gae = T_A_gae + T_V_curr

    return T_Q_gae, T_A_gae


class MovingAverage(NamedTuple):
    mean: FloatScalar
    var: FloatScalar
    count: IntScalar

    @classmethod
    def create(cls):
        return MovingAverage(0.0, 1.0, 0)

    def update(self, arr):
        assert arr.ndim == 1
        batch_mean = jnp.mean(arr)
        batch_var = jnp.var(arr)
        batch_count = len(arr)
        return self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: FloatScalar, batch_var: FloatScalar, batch_count: float):
        delta_mean = batch_mean - self.mean
        tot_count = self.count + batch_count

        rate = batch_count / tot_count
        new_mean = self.mean + delta_mean * rate
        new_var = self.var + rate * (batch_var - self.var + delta_mean * (batch_mean - new_mean))

        return MovingAverage(new_mean, new_var, tot_count)

    @property
    def std(self):
        return jnp.sqrt(jnp.maximum(self.var, 1e-9))

    def unnormalize(self, normalized):
        return normalized * self.std + self.mean

    def normalize(self, unnormalized):
        return (unnormalized - self.mean) / self.std


class PPOAlg(struct.PyTreeNode):
    update_idx: int
    key: PRNGKey
    policy: TrainState[tfd.Distribution]
    V: TrainState[FloatScalar]
    V_ma: MovingAverage

    # Adaptive learning rate for the policy.
    pol_lr: FloatScalar

    # In units of update_idx.
    ent_cf_sched: optax.Schedule = struct.field(pytree_node=False)

    env: Env = struct.field(pytree_node=False)
    cfg: PPOCfg = struct.field(pytree_node=False)

    @classmethod
    def create(cls, key: PRNGKey, env: Env, cfg: PPOCfg):
        key, key_pol, key_V = jr.split(key, 3)

        act = get_act_from_str(cfg.act)

        obs, obs_priv, state = env.reset(key)

        # Define policy network.
        pol_base_cls = ft.partial(MLP, cfg.pol_hid_sizes, act, act_final=True)
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
        V_def = ValueNet(V_base_cls)
        V_tx = _optim(cfg.val_lr, cfg.clip_grad)
        V = TrainState.create_from_def(key_V, V_def, (obs_priv,), V_tx)

        V_ma = MovingAverage.create()
        ent_cf_sched = as_schedule(cfg.entropy_cf).make()
        return PPOAlg(0, key, pol, V, V_ma, cfg.pol_lr, ent_cf_sched, env, cfg)

    def make_dset(self, b_data: RolloutOutput) -> PPOBatch:
        # 1: Compute V from data.
        bTp1_V = jax_vmap(self.get_V, rep=2)(b_data.Tp1_obs_priv)

        # 2: Compute Q using GAE.
        compute_gae_ = ft.partial(compute_gae, self.cfg.disc_gamma, self.cfg.train_cfg.gae_lambda)
        bT_Q_gae, bT_A_gae = jax.vmap(compute_gae_)(b_data.T_reward, bTp1_V, b_data.T_terminated)

        batch_size, T = bT_Q_gae.shape
        bT_obs, bT_obspriv, bT_V = b_data.Tp1_obs[:, :-1], b_data.Tp1_obs_priv[:, :-1], bTp1_V[:, :-1]

        # 3: Make the dataset by flattening (b, T) -> (b * T, )
        bT_dset = PPOBatch(bT_obs, bT_obspriv, b_data.T_action, b_data.T_logprob, bT_A_gae, bT_Q_gae)
        b_dset = jtu.tree_map(lambda x: merge01(x), bT_dset)
        return b_dset

    @ft.partial(jax.jit, donate_argnums=1)
    def collect(self, collector: Collector) -> tuple[Collector, RolloutOutput]:
        return collector.collect_batch(self.policy.apply)

    @property
    def ent_cf(self):
        return self.ent_cf_sched(self.update_idx)

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, b_data: RolloutOutput):
        train_cfg = self.cfg.train_cfg

        # First, rescale the rewards.
        b_data = b_data._replace(T_reward=b_data.T_reward / self.cfg.rew_scale)

        def updates_body_outer(alg__: PPOAlg, _):
            def updates_body(alg_: PPOAlg, batch: PPOBatch):
                # Only normalize targets for value update.
                alg_, val_info = alg_.update_value(batch)
                alg_, pol_info = alg_.update_policy(batch)
                return alg_, {**val_info, **pol_info}

            # 1: Make dataset, i.e., recompute advantages.
            b_dset = alg__.make_dset(b_data)
            dataset_size = len(b_dset.obs)
            assert dataset_size % train_cfg.batch_size == 0
            n_batches = dataset_size // train_cfg.batch_size

            # 2: Update MA for V.
            if self.cfg.train_cfg.normalize_V:
                V_ma = alg__.V_ma.update(b_dset.q)
                alg__ = alg__.replace(V_ma=V_ma)

            # 3: Shuffle and reshape
            key_shuffle = jr.fold_in(self.key, alg__.V.step)
            rand_idxs = jr.permutation(key_shuffle, jnp.arange(dataset_size))
            b_dset = jtu.tree_map(lambda x: x[rand_idxs], b_dset)
            mb_dset: PPOBatch = tree_split_dims(b_dset, (n_batches, train_cfg.batch_size))

            # 4: Perform value function and policy updates.
            alg__, info_ = lax.scan(updates_body, alg__, mb_dset, length=n_batches)
            # Mean over batches.
            info_ = jtu.tree_map(lambda x: x.mean(), info_)
            return alg__, info_

        new_self, info = lax.scan(updates_body_outer, self, None, length=self.cfg.train_cfg.n_update_epochs)

        new_self = new_self.replace(update_idx=self.update_idx + 1)

        # Mean over batches.
        info = jtu.tree_map(lambda x: x.mean(), info)
        info["mean_uprob"] = jnp.mean(jnp.exp(b_data.T_logprob))
        info["V_ma/mean"] = new_self.V_ma.mean
        info["V_ma/std"] = new_self.V_ma.std
        info["Anneal/pol_lr"] = new_self.policy.opt_state.hyperparams["learning_rate"]

        return new_self, info

    def get_V(self, obs_priv: Obs, params=None) -> FloatScalar:
        if params is None:
            params = self.V.params
        V_norm = self.V.apply_with(obs_priv, params=params)
        return self.V_ma.unnormalize(V_norm)

    def update_value(self, batch: PPOBatch) -> tuple["PPOAlg", MetricsDict]:
        def get_val_loss(V_params):
            V_apply = ft.partial(self.V.apply_with, params=V_params)
            b_V_norm = jax.vmap(V_apply)(batch.obs_priv)
            # Define the loss on the normalized value function.
            b_loss_mse = (b_V_norm - b_Q_norm) ** 2
            loss_mse = b_loss_mse.mean()
            info_ = {"loss_V": loss_mse}
            return loss_mse, info_

        b_Q_norm = self.V_ma.normalize(batch.q)

        grads_V, val_info = jax.grad(get_val_loss, has_aux=True)(self.V.params)
        V = self.V.apply_gradients(grads=grads_V)
        val_info["V_grad"] = compute_norm(grads_V)
        return self.replace(V=V), val_info

    def update_policy(self, batch: PPOBatch) -> tuple["PPOAlg", MetricsDict]:
        def get_pol_loss(pol_params):
            def get_logprob_entropy(obs, control, ent_key):
                dist = self.policy.apply_with(obs, params=pol_params)
                if self.cfg.pol_type == "Normal":
                    entropy = dist.entropy()
                elif self.cfg.pol_type == "TanhNormal":
                    entropy = dist.entropy(seed=ent_key)
                else:
                    raise NotImplementedError("")
                return dist.log_prob(control), entropy

            b_logprobs, b_entropy = jax_vmap(get_logprob_entropy)(batch.obs, batch.control, b_ent_key)
            b_logratios = b_logprobs - batch.logprob
            is_ratio = jnp.exp(b_logratios)
            adv = batch.advantage

            pg_loss_orig = -adv * is_ratio
            pg_loss_clip = -adv * jnp.clip(is_ratio, 1 - clip_ratio, 1 + clip_ratio)
            loss_pg = jnp.maximum(pg_loss_orig, pg_loss_clip).mean()
            pol_clipfrac = jnp.mean(pg_loss_clip > pg_loss_orig)

            mean_entropy = b_entropy.mean()
            loss_entropy = -mean_entropy

            # Compute KL between old and new policy. Adjust lr if needed.
            #   KL( pi_old || pi_new )
            b_logprob_old = batch.logprob
            b_logprob_new = b_logprobs
            kl_old_new_ = jnp.mean(b_logprob_old - b_logprob_new)

            pol_loss = loss_pg + self.ent_cf * loss_entropy
            info = {
                "loss_pol": pol_loss,
                "entropy": mean_entropy,
                "pol_clipfrac": pol_clipfrac,
                "kl_old_new": kl_old_new_,
            }
            return pol_loss, info

        clip_ratio = self.cfg.train_cfg.clip_ratio
        ent_key0 = jr.fold_in(self.key, self.policy.step)
        b_ent_key = jr.split(ent_key0, len(batch.obs))
        grads, pol_info = jax.grad(get_pol_loss, has_aux=True)(self.policy.params)

        # Adjust lr if needed.
        kl_old_new = pol_info["kl_old_new"]
        kl_desired = self.cfg.train_cfg.kl_desired
        kl_too_high = kl_old_new > kl_desired * 2
        kl_too_low = (kl_old_new < kl_desired / 2) & (kl_old_new > 0.0)

        lr_lower = jnp.maximum(self.cfg.train_cfg.pol_lr_min, self.pol_lr / 1.5)
        lr_higher = jnp.minimum(self.cfg.train_cfg.pol_lr_max, self.pol_lr * 1.5)
        pol_lr_new = jnp.where(kl_too_high, lr_lower, jnp.where(kl_too_low, lr_higher, self.pol_lr))
        self.policy.opt_state.hyperparams["learning_rate"] = pol_lr_new

        grads, pol_info["pol_grad"] = compute_norm_and_clip(grads, self.cfg.clip_grad)
        policy = self.policy.apply_gradients(grads=grads)
        # ipdb.set_trace()
        return self.replace(policy=policy, pol_lr=pol_lr_new), pol_info

    @jax.jit
    def estimate_tv(self, b_data: RolloutOutput) -> FloatScalar:
        def get_logprob(obs, control):
            return self.policy.apply(obs).log_prob(control)

        # bT_obs = jax_vmap(jax_vmap(self.task.get_obs, in_axes=(0, None)))(data.bTp1_x[:, :-1, :], data.b_param)
        bT_obs = b_data.Tp1_obs[:, :-1]
        bT_logp_behavior = b_data.T_logprob
        bT_logp_now = jax_vmap(get_logprob, rep=2)(bT_obs, b_data.T_action)
        assert bT_logp_now.shape == bT_logp_behavior.shape

        is_ratio = jnp.exp(bT_logp_now - bT_logp_behavior)
        return 0.5 * jnp.mean(jnp.abs(is_ratio - 1.0))

    def act(self, obs: Obs) -> Control:
        """Convenience function for deployment."""
        return self.policy.apply(obs).mode()

    @jax.jit
    def eval(self, collector: Collector) -> PPOEval:
        def eval_pol(obs):
            dist = self.policy.apply(obs)
            std_approx = _get_std(dist)
            info = {"std": std_approx}
            # info = {"std": dist.stddev()}
            return dist.mode(), info

        bT_state, bT_out, bT_control, bT_misc = collector.rollout_eval(eval_pol)
        batch_size, T, _ = bT_control.shape

        # Compute the mask.
        bT_invalid = lax.cummax(1 * bT_out.terminated, axis=1) > 0
        bT_valid_mask = jnp.concatenate([jnp.ones((batch_size, 1), dtype=bool), ~bT_invalid[:, :-1]], axis=1)
        assert bT_valid_mask.dtype == bool

        ppo_info = {
            "Steps/Policy": self.policy.step,
            "Steps/V": self.V.step,
            "Steps/Collect": collector.collect_idx,
            "Steps/Update": self.update_idx,
            "Anneal/DiscGamma": self.cfg.disc_gamma,
            "Anneal/EntCf": self.ent_cf,
        }

        return PPOEval(bT_valid_mask, bT_state, bT_control, bT_out.reward, bT_out.info, bT_misc, ppo_info)


def _get_std(dist: tfd.Distribution):
    assert isinstance(dist, tfd.Independent)
    inner_dist = dist.distribution
    if isinstance(inner_dist, tfd.Normal):
        return inner_dist.stddev()
    elif isinstance(inner_dist, TanhTransformedDistribution):
        inner_dist2 = inner_dist.distribution
        assert isinstance(inner_dist2, tfd.Normal)
        return inner_dist2.stddev()
    else:
        raise NotImplementedError("")
