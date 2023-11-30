from typing import Type

import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from flax.linen import initializers

from jaxrl.networks.network_utils import default_nn_init, scaled_init
from jaxrl.utils.jax_types import AnyFloat
from jaxrl.utils.tfp import tfb, tfd


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, threshold: float = 0.999, validate_args: bool = False):
        super().__init__(distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args)
        self._threshold = threshold
        self.inverse_threshold = self.bijector.inverse(threshold)

        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = np.log(1.0 - threshold)

        self._log_prob_left = self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        self._log_prob_right = self.distribution.log_survival_function(inverse_threshold) - log_epsilon

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(event >= self._threshold, self._log_prob_right, super().log_prob(event)),
        )

    def entropy(self, seed=None):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    def _mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


def inv_softplus(std: float) -> float:
    return np.log(np.expm1(std))


class NormalPolicyStdConst(nn.Module):
    base_cls: Type[nn.Module]
    _nu: int
    std_init: float = 1.0

    @nn.compact
    def __call__(self, obs: AnyFloat, scale_final: float = 1e-2, *args, **kwargs) -> tfd.Distribution:
        nn_init = default_nn_init()
        x = self.base_cls()(obs, *args, **kwargs)

        means = nn.Dense(self._nu, kernel_init=scaled_init(nn_init, scale_final), name="DenseMean")(x)
        stds = self.param("std", initializers.constant(self.std_init), (self._nu,), means.dtype)
        assert means.shape == stds.shape

        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(distribution, reinterpreted_batch_ndims=1)


class NormalPolicyStdVary(nn.Module):
    base_cls: Type[nn.Module]
    _nu: int
    std_min: float = 1e-9
    std_init: float = 0.5

    @nn.compact
    def __call__(self, obs: AnyFloat, scale_final: float = 1e-2, *args, **kwargs) -> tfd.Distribution:
        nn_init = default_nn_init()
        x = self.base_cls()(obs, *args, **kwargs)

        means = nn.Dense(self._nu, kernel_init=scaled_init(nn_init, scale_final), name="DenseMean")(x)

        stds_raw = nn.Dense(self._nu, kernel_init=scaled_init(nn_init, scale_final), name="DenseStdRaw")(x)
        std_raw_offset = inv_softplus(self.std_init)
        stds = jnn.softplus(stds_raw + std_raw_offset) + self.std_min

        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(distribution, reinterpreted_batch_ndims=1)


class TanhNormalPolicy(nn.Module):
    base_cls: Type[nn.Module]
    _nu: int
    std_min: float = 0.0
    std_init: float = 0.5

    @nn.compact
    def __call__(self, obs: AnyFloat, scale_final: float = 1e-2, *args, **kwargs) -> tfd.Distribution:
        nn_init = default_nn_init()
        x = self.base_cls()(obs, *args, **kwargs)

        means = nn.Dense(self._nu, kernel_init=scaled_init(nn_init, scale_final), name="DenseMean")(x)

        stds_raw = nn.Dense(self._nu, kernel_init=scaled_init(nn_init, scale_final), name="DenseStdRaw")(x)
        std_raw_offset = inv_softplus(self.std_init)
        stds = jnn.softplus(stds_raw + std_raw_offset) + self.std_min

        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(TanhTransformedDistribution(distribution, threshold=0.999), reinterpreted_batch_ndims=1)
