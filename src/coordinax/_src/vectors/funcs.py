"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__: list[str] = []

import functools as ft
from inspect import isclass
from typing import cast

import equinox as eqx
from jaxtyping import Array, Shaped
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u

from . import api
from coordinax._src.vectors.base import AbstractVector


@dispatch
@ft.partial(eqx.filter_jit, inline=True)
def normalize_vector(x: Shaped[Array, "*batch N"], /) -> Shaped[Array, "*batch N"]:
    """Return the unit vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> x = jnp.asarray([2, 0])
    >>> cx.vecs.normalize_vector(x)
    Array([1., 0.], dtype=float32)

    >>> x = jnp.asarray([0, 2])
    >>> cx.vecs.normalize_vector(x)
    Array([0., 1.], dtype=float32)

    """
    return x / jnp.linalg.vector_norm(x, axis=-1, keepdims=True)


@dispatch
@ft.partial(eqx.filter_jit, inline=True)
def normalize_vector(
    x: Shaped[u.AbstractQuantity, "*batch N"], /
) -> Shaped[u.AbstractQuantity, "*batch N"]:
    """Return the unit vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = u.Quantity(jnp.asarray([2, 0]), "km")
    >>> cx.vecs.normalize_vector(x)
    Quantity(Array([1., 0.], dtype=float32), unit='')

    >>> x = u.Quantity(jnp.asarray([0, 2]), "s")
    >>> cx.vecs.normalize_vector(x)
    Quantity(Array([0., 1.], dtype=float32), unit='')

    """
    return x / jnp.linalg.vector_norm(x, axis=-1, keepdims=True)


# ===========================================================================


@dispatch
def time_nth_derivative_vector_type(
    obj: type[AbstractVector] | AbstractVector, /, *, n: int
) -> type[AbstractVector]:
    out = cast(type[AbstractVector], obj if isclass(obj) else type(obj))
    if n == 0:
        pass
    elif n < 0:
        for _ in range(-n):
            out = api.time_antiderivative_vector_type(out)
    else:
        for _ in range(n):
            out = api.time_derivative_vector_type(out)

    return out
