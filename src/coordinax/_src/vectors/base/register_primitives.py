"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

import jax
from quax import quaxify, register

import quaxed.numpy as jnp
from dataclassish import field_items, replace

from .base import AbstractVector


@register(jax.lax.convert_element_type_p)  # type: ignore[misc]
def _convert_element_type_p(operand: AbstractVector, **kwargs: Any) -> AbstractVector:
    """Convert the element type of a quantity."""
    # TODO: examples
    convert_p = quaxify(jax.lax.convert_element_type_p.bind)
    return replace(
        operand,
        **{k: convert_p(v, **kwargs) for k, v in field_items(operand)},
    )


@register(jax.lax.broadcast_in_dim_p)  # type: ignore[misc]
def _broadcast_in_dim_p(
    operand: AbstractVector, *, shape: tuple[int, ...], **kwargs: Any
) -> AbstractVector:
    """Broadcast in a dimension.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    Cartesian 1D position, velocity, and acceleration:

    >>> q = cx.vecs.CartesianPos1D.from_([1], "m")
    >>> q.x
    Quantity['length'](Array(1., dtype=float32), unit='m')

    >>> jnp.broadcast_to(q, (1, 1)).x
    Quantity['length'](Array([1.], dtype=float32), unit='m')

    >>> p = cx.vecs.CartesianVel1D.from_([1], "m/s")
    >>> p.d_x
    Quantity['speed'](Array(1, dtype=int32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 1)).d_x
    Quantity['speed'](Array([1], dtype=int32), unit='m / s')

    >>> a = cx.vecs.CartesianAcc1D.from_([1], "m/s2")
    >>> a.d2_x
     Quantity['acceleration'](Array(1, dtype=int32), unit='m / s2')

    >>> jnp.broadcast_to(a, (1, 1)).d2_x
    Quantity['acceleration'](Array([1], dtype=int32), unit='m / s2')


    Radial 1D position, velocity, and acceleration:

    >>> q = cx.vecs.RadialPos.from_([1], "m")
    >>> q.r
    Distance(Array(1., dtype=float32), unit='m')

    >>> jnp.broadcast_to(q, (1, 1)).r
    Distance(Array([1.], dtype=float32), unit='m')

    >>> p = cx.vecs.RadialVel.from_([1], "m/s")
    >>> p.d_r
    Quantity['speed'](Array(1, dtype=int32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 1)).d_r
    Quantity['speed'](Array([1], dtype=int32), unit='m / s')

    >>> a = cx.vecs.RadialAcc.from_([1], "m/s2")
    >>> a.d2_r
    Quantity['acceleration'](Array(1, dtype=int32), unit='m / s2')

    >>> jnp.broadcast_to(a, (1, 1)).d2_r
    Quantity['acceleration'](Array([1], dtype=int32), unit='m / s2')


    Cartesian 2D position, velocity, and acceleration:

    >>> q = cx.vecs.CartesianPos2D.from_([1, 2], "m")
    >>> q.x
    Quantity['length'](Array(1., dtype=float32), unit='m')

    >>> jnp.broadcast_to(q, (1, 2)).x
    Quantity['length'](Array([1.], dtype=float32), unit='m')

    >>> p = cx.vecs.CartesianVel2D.from_([1, 2], "m/s")
    >>> p.d_x
    Quantity['speed'](Array(1., dtype=float32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 2)).d_x
    Quantity['speed'](Array([1.], dtype=float32), unit='m / s')

    >>> a = cx.vecs.CartesianAcc2D.from_([1, 2], "m/s2")
    >>> a.d2_x
    Quantity['acceleration'](Array(1., dtype=float32), unit='m / s2')

    >>> jnp.broadcast_to(a, (1, 2)).d2_x
    Quantity['acceleration'](Array([1.], dtype=float32), unit='m / s2')


    Cartesian 3D position, velocity, and acceleration:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> q.x
    Quantity['length'](Array(1., dtype=float32), unit='m')

    >>> jnp.broadcast_to(q, (1, 3)).x
    Quantity['length'](Array([1.], dtype=float32), unit='m')

    >>> p = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
    >>> p.d_x
    Quantity['speed'](Array(1., dtype=float32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 3)).d_x
    Quantity['speed'](Array([1.], dtype=float32), unit='m / s')

    >>> a = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "m/s2")
    >>> a.d2_x
    Quantity['acceleration'](Array(1., dtype=float32), unit='m / s2')

    >>> jnp.broadcast_to(a, (1, 3)).d2_x
     Quantity['acceleration'](Array([1.], dtype=float32), unit='m / s2')

    """
    # TODO: use `jax.lax.broadcast_in_dim_p`
    c_shape = shape[:-1]
    return replace(
        operand,
        **{k: jnp.broadcast_to(v, c_shape) for k, v in field_items(operand)},
    )
