from typing import Type

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from jaxrl.utils.jax_types import AnyFloat


class TVEncoder(nn.Module):
    enc_cls: Type[nn.Module]
    mlp: Type[nn.Module]
    time_min: float
    time_max: float
    n_freqs: int

    @nn.compact
    def __call__(self, obs: AnyFloat) -> AnyFloat:
        time, other = obs[..., 0], obs[..., 1:]
        assert time.shape == tuple()

        alpha = 0.95
        scale_min = alpha * (1 / self.time_max) * np.pi / 2
        # Scaled Nyquist.
        scale_max = 0.6 * (1 / self.time_min) * np.pi
        scale_factor = np.log(scale_max / scale_min) / (self.n_freqs - 1)
        div_term = scale_min * np.exp(np.arange(0, self.n_freqs) * scale_factor)

        sin_feat = jnp.sin(div_term * time)
        cos_feat = jnp.cos(div_term * time)

        sincos_feat = jnp.concatenate([sin_feat, cos_feat], axis=-1)
        #############################################################
        # 1: Pass through encoder.
        time_feat = self.enc_cls()(sincos_feat)
        # time_feat = sincos_feat

        # 2: Concatenate with other features, pass through MLP.
        feat = jnp.concatenate([time_feat, other], axis=-1)
        out = self.mlp()(feat)
        return out
