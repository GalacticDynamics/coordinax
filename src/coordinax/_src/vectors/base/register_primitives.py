"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any, TypeAlias

import jax
from jax import tree as jtu
from jaxtyping import Array, Bool
from quax import quaxify, register

import quaxed.numpy as jnp
import unxt.quantity as uq
from dataclassish import field_items, replace

from .vector import AbstractVector

# ===================================================================

Shape: TypeAlias = tuple[int, ...]


@register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p(
    operand: AbstractVector, *, shape: Shape, **kwargs: Any
) -> AbstractVector:
    """Broadcast in a dimension.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    Cartesian 1D position, velocity, and acceleration:

    >>> q = cx.vecs.CartesianPos1D.from_([1], "m")
    >>> q.x
    Quantity['length'](Array(1, dtype=int32), unit='m')

    >>> jnp.broadcast_to(q, (1, 1)).x
    Quantity['length'](Array([1], dtype=int32), unit='m')

    >>> p = cx.vecs.CartesianVel1D.from_([1], "m/s")
    >>> p.x
    Quantity['speed'](Array(1, dtype=int32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 1)).x
    Quantity['speed'](Array([1], dtype=int32), unit='m / s')

    >>> a = cx.vecs.CartesianAcc1D.from_([1], "m/s2")
    >>> a.x
     Quantity['acceleration'](Array(1, dtype=int32), unit='m / s2')

    >>> jnp.broadcast_to(a, (1, 1)).x
    Quantity['acceleration'](Array([1], dtype=int32), unit='m / s2')


    Radial 1D position, velocity, and acceleration:

    >>> q = cx.vecs.RadialPos.from_([1], "m")
    >>> q.r
    Distance(Array(1, dtype=int32), unit='m')

    >>> jnp.broadcast_to(q, (1, 1)).r
    Distance(Array([1], dtype=int32), unit='m')

    >>> p = cx.vecs.RadialVel.from_([1], "m/s")
    >>> p.r
    Quantity['speed'](Array(1, dtype=int32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 1)).r
    Quantity['speed'](Array([1], dtype=int32), unit='m / s')

    >>> a = cx.vecs.RadialAcc.from_([1], "m/s2")
    >>> a.r
    Quantity['acceleration'](Array(1, dtype=int32), unit='m / s2')

    >>> jnp.broadcast_to(a, (1, 1)).r
    Quantity['acceleration'](Array([1], dtype=int32), unit='m / s2')


    Cartesian 2D position, velocity, and acceleration:

    >>> q = cx.vecs.CartesianPos2D.from_([1, 2], "m")
    >>> print(q)
    <CartesianPos2D (x[m], y[m])
        [1 2]>

    >>> print(jnp.broadcast_to(q, (1, 2)))
    <CartesianPos2D (x[m], y[m])
        [[1 2]]>

    >>> p = cx.vecs.CartesianVel2D.from_([1, 2], "m/s")
    >>> print(p)
    <CartesianVel2D (x[m / s], y[m / s])
        [1 2]>

    >>> print(jnp.broadcast_to(p, (1, 2)))
    <CartesianVel2D (x[m / s], y[m / s])
        [[1 2]]>

    >>> a = cx.vecs.CartesianAcc2D.from_([1, 2], "m/s2")
    >>> print(a)
    <CartesianAcc2D (x[m / s2], y[m / s2])
        [1 2]>

    >>> print(jnp.broadcast_to(a, (1, 2)))
    <CartesianAcc2D (x[m / s2], y[m / s2])
        [[1 2]]>

    Cartesian 3D position, velocity, and acceleration:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> q.x
    Quantity['length'](Array(1, dtype=int32), unit='m')

    >>> jnp.broadcast_to(q, (1, 3)).x
    Quantity['length'](Array([1], dtype=int32), unit='m')

    >>> p = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
    >>> p.x
    Quantity['speed'](Array(1, dtype=int32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 3)).x
    Quantity['speed'](Array([1], dtype=int32), unit='m / s')

    >>> a = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "m/s2")
    >>> print(a)
    <CartesianAcc3D (x[m / s2], y[m / s2], z[m / s2])
        [1 2 3]>

    >>> print(jnp.broadcast_to(a, (1, 3)))
    <CartesianAcc3D (x[m / s2], y[m / s2], z[m / s2])
        [[1 2 3]]>

    """
    # TODO: use `jax.lax.broadcast_in_dim_p`
    c_shape = shape[:-1]
    return replace(
        operand,
        **{k: jnp.broadcast_to(v, c_shape) for k, v in field_items(operand)},
    )


# ===================================================================


@register(jax.lax.convert_element_type_p)
def convert_element_type_p(operand: AbstractVector, **kwargs: Any) -> AbstractVector:
    """Convert the element type of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPosND.from_([1, 2, 3], "m")
    >>> qlax.convert_element_type(vec, float)
    CartesianPosND(q=Quantity[...](value=f32[3], unit=Unit("m")))

    """
    convert_p = quaxify(jax.lax.convert_element_type_p.bind)
    return replace(
        operand,
        **{k: convert_p(v, **kwargs) for k, v in field_items(operand)},
    )


# ===================================================================


@register(jax.lax.eq_p)
def eq_vec_vec(lhs: AbstractVector, rhs: AbstractVector, /) -> Bool[Array, "..."]:
    """Element-wise equality of two vectors.

    See `AbstractVector.__eq__` for examples.

    """
    # Map the equality over the leaves, which are Quantities.
    comp_tree = jtu.map(
        jnp.equal,
        jtu.leaves(lhs, is_leaf=uq.is_any_quantity),
        jtu.leaves(rhs, is_leaf=uq.is_any_quantity),
        is_leaf=uq.is_any_quantity,
    )

    # Reduce the equality over the leaves.
    return jax.tree.reduce(jnp.logical_and, comp_tree)
