from typing import NamedTuple, Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Shaped

PRNGKey = PRNGKeyArray

Arr = Union[np.ndarray, Array]

AnyShaped = Shaped[Arr, "*"]
AnyFloat = Float[Arr, "*"]
Shape = tuple[int, ...]

Vec2 = Float[Arr, "2"]
Vec3 = Float[Arr, "3"]

BVec2 = Float[Arr, "b 2"]
BVec3 = Float[Arr, "b 3"]

BBVec3 = Float[Arr, "b1 b2 3"]

FloatScalar = float | Float[Arr, ""]
IntScalar = int | Int[Arr, ""]
BoolScalar = bool | Bool[Arr, ""]

BFloat = Float[Arr, "b"]
BInt = Int[Arr, "b"]
BBool = Bool[Arr, "b"]

BBFloat = Float[Arr, "b1 b2"]
BBBool = Bool[Arr, "b1 b2"]
BBInt = Int[Arr, "b1 b2"]

TFloat = Float[Arr, "b T"]
Tp1Float = Float[Arr, "b Tp1"]

TBool = Bool[Arr, "b"]

BTFloat = Float[Arr, "b T"]
BTInt = Int[Arr, "b T"]
BTBool = Bool[Arr, "b T"]

MetricsDict = dict[str, Union[FloatScalar, "MetricsDict"]]
