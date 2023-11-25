from typing import Type

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Float

from jaxrl.networks.network_utils import default_nn_init
from jaxrl.utils.jax_types import Arr
from jaxrl.utils.shape_utils import assert_shape


class ValueNet(nn.Module):
    net_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, state: Float[Arr, "* nx"], *args, **kwargs) -> Float[Arr, "*"]:
        batch_shape = state.shape[:-1]
        x = self.net_cls()(state, *args, **kwargs)
        Vl = nn.Dense(1, kernel_init=default_nn_init())(x)
        return assert_shape(Vl.squeeze(-1), batch_shape)
