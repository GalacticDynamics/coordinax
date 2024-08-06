"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = [
    "represent_as",
    "normalize_vector",
]

from functools import partial
from typing import Any

import jax
from jaxtyping import Array, Shaped
from plum import dispatch

import quaxed.array_api as xp
from unxt import Quantity


@dispatch.abstract  # type: ignore[misc]
def represent_as(current: Any, target: type[Any], /, **kwargs: Any) -> Any:
    """Transform the current vector to the target vector."""
    raise NotImplementedError  # pragma: no cover


# ===================================================================


@dispatch
@partial(jax.jit, inline=True)
def normalize_vector(x: Shaped[Array, "*batch N"], /) -> Shaped[Array, "*batch N"]:
    """Return the unit vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> x = jnp.asarray([2, 0])
    >>> cx.normalize_vector(x)
    Array([1., 0.], dtype=float64)

    >>> x = jnp.asarray([0, 2])
    >>> cx.normalize_vector(x)
    Array([0., 1.], dtype=float64)

    """
    return x / xp.linalg.vector_norm(x, axis=-1, keepdims=True)


@dispatch
@partial(jax.jit, inline=True)
def normalize_vector(
    x: Shaped[Quantity, "*batch N"], /
) -> Shaped[Quantity, "*batch N"]:
    """Return the unit vector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = Quantity(jnp.asarray([2, 0]), "km")
    >>> cx.normalize_vector(x)
    Quantity['dimensionless'](Array([1., 0.], dtype=float64), unit='')

    >>> x = Quantity(jnp.asarray([0, 2]), "s")
    >>> cx.normalize_vector(x)
    Quantity['dimensionless'](Array([0., 1.], dtype=float64), unit='')

    """
    return x / xp.linalg.vector_norm(x, axis=-1, keepdims=True)
