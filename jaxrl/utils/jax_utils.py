import functools as ft
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar

import einops as ei
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Float

from jaxrl.utils.jax_types import Arr, Shape

_PyTree = TypeVar("_PyTree")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


def rep_vmap(fn: _Fn, rep: int, in_axes: int | Sequence[Any] = 0, **kwargs) -> _Fn:
    for ii in range(rep):
        fn = jax.vmap(fn, in_axes=in_axes, **kwargs)
    return fn


def jax_vmap(fn: _Fn, in_axes: int | Sequence[Any] = 0, out_axes: Any = 0, rep: int = None) -> _Fn:
    if rep is not None:
        return rep_vmap(fn, rep=rep, in_axes=in_axes, out_axes=out_axes)

    return jax.vmap(fn, in_axes, out_axes)


def tree_split_dims(tree: _PyTree, new_dims: Shape) -> _PyTree:
    prod_dims = np.prod(new_dims)

    def tree_split_dims_inner(arr: Arr) -> Arr:
        assert arr.shape[0] == prod_dims
        target_shape = new_dims + arr.shape[1:]
        return arr.reshape(target_shape)

    return jtu.tree_map(tree_split_dims_inner, tree)


def concat_at_front(arr1: Float[Arr, "nx"], arr2: Float[Arr, "T nx"], axis: int = 0) -> Float[Arr, "Tp1 nx"]:
    """
    :param arr1: (nx, )
    :param arr2: (T, nx)
    :param axis: Which axis for arr1 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr2_shape = list(arr2.shape)
    del arr2_shape[axis]
    assert np.all(np.array(arr1.shape) == np.array(arr2_shape))

    return jnp.concatenate([jnp.expand_dims(arr1, axis=axis), arr2], axis=axis)
