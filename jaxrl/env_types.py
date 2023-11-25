from jaxtyping import Float

from jaxrl.utils.jax_types import Arr

State = Float[Arr, "nxx"]
Control = Float[Arr, "nu"]
Obs = Float[Arr, "obs"]

BState = Float[Arr, "b nx"]
BObs = Float[Arr, "b nobs"]

TObs = Float[Arr, "b nobs"]
TControl = Float[Arr, "b nu"]

BTState = Float[Arr, "b T nx"]
BTControl = Float[Arr, "b T nu"]
