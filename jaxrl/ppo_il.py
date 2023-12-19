import functools as ft
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from flax import struct

from jaxrl.env import Env
from jaxrl.env_types import Control, Obs, State
from jaxrl.ppo import PPOAlg, PPOCfg, RolloutOutput, compute_gae
from jaxrl.utils.grad_utils import compute_norm_and_clip
from jaxrl.utils.jax_types import FloatScalar, MetricsDict, PRNGKey
from jaxrl.utils.jax_utils import jax_vmap, merge01


class PPOILBatch(NamedTuple):
    obs: Obs
    obs_priv: Obs
    control: Control
    logprob: FloatScalar
    advantage: FloatScalar
    q: FloatScalar


class PPOILAlg(PPOAlg):
    pol_expert: Callable[[Any], Control] = struct.field(pytree_node=False)
    bc_coeff: float = struct.field(pytree_node=True)

    @classmethod
    def create(cls, key: PRNGKey, env: Env, cfg: PPOCfg, pol_expert, bc_coeff: float):
        ppo_alg = PPOAlg.create(key, env, cfg)
        return cls(
            ppo_alg.update_idx,
            ppo_alg.key,
            ppo_alg.policy,
            ppo_alg.V,
            ppo_alg.V_ma,
            ppo_alg.pol_lr,
            ppo_alg.ent_cf_sched,
            ppo_alg.env,
            ppo_alg.cfg,
            pol_expert,
            bc_coeff,
        )

    def update_policy(self, batch: PPOILBatch) -> tuple["PPOAlg", MetricsDict]:
        def get_pol_loss(pol_params):
            def get_logprob_entropy(obs, control, ent_key):
                dist = self.policy.apply_with(obs, params=pol_params)
                if self.cfg.pol_type == "Normal":
                    entropy = dist.entropy()
                elif self.cfg.pol_type == "TanhNormal":
                    entropy = dist.entropy(seed=ent_key)
                else:
                    raise NotImplementedError("")
                return dist.log_prob(control), entropy, dist.mode()

            b_logprobs, b_entropy, b_mode = jax_vmap(get_logprob_entropy)(batch.obs, batch.control, b_ent_key)
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

            # BC loss with the expert policy.
            b_loss_bc = jnp.sum((b_mode - b_mode_expert) ** 2, axis=-1)
            loss_bc = jnp.mean(b_loss_bc)

            pol_loss = loss_pg + self.ent_cf * loss_entropy + self.bc_coeff * loss_bc
            info = {
                "loss_pol": pol_loss,
                "entropy": mean_entropy,
                "pol_clipfrac": pol_clipfrac,
                "kl_old_new": kl_old_new_,
                "loss_bc": loss_bc,
            }
            return pol_loss, info

        b_mode_expert = jax_vmap(self.pol_expert)(batch.obs_priv)

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
