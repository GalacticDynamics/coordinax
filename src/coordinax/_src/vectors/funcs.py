"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__: list[str] = []

from functools import partial

import equinox as eqx
from jaxtyping import Array, Shaped
from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import AbstractQuantity


@dispatch
@partial(eqx.filter_jit, inline=True)
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
@partial(eqx.filter_jit, inline=True)
def normalize_vector(
    x: Shaped[AbstractQuantity, "*batch N"], /
) -> Shaped[AbstractQuantity, "*batch N"]:
    """Return the unit vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = u.Quantity(jnp.asarray([2, 0]), "km")
    >>> cx.vecs.normalize_vector(x)
    Quantity['dimensionless'](Array([1., 0.], dtype=float32), unit='')

    >>> x = u.Quantity(jnp.asarray([0, 2]), "s")
    >>> cx.vecs.normalize_vector(x)
    Quantity['dimensionless'](Array([0., 1.], dtype=float32), unit='')

    """
    return x / jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
