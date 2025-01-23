"""Representation of coordinates in different systems."""

__all__: list[str] = []

from dataclasses import replace
from typing import Any, cast

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

import quaxed.lax as qlax
from unxt.quantity import AbstractQuantity

from .cartesian import CartesianPosND
from coordinax._src.vectors.base_pos import AbstractPos

# ------------------------------------------------


@register(jax.lax.add_p)
def add_vcnd(lhs: CartesianPosND, rhs: AbstractPos, /) -> CartesianPosND:
    """Add two vectors.

    Examples
    --------
    >>> import coordinax as cx

    A 3D vector:

    >>> q1 = cx.vecs.CartesianPosND.from_([1, 2, 3], "km")
    >>> q2 = cx.vecs.CartesianPosND.from_([2, 3, 4], "km")
    >>> (q1 + q2).q
    Quantity['length'](Array([3, 5, 7], dtype=int32), unit='km')

    """
    cart = cast(CartesianPosND, rhs.vconvert(CartesianPosND))
    return replace(lhs, q=lhs.q + cart.q)


# ------------------------------------------------
# Dot product
# TODO: see implementation in https://github.com/google/tree-math for how to do
# this more generally.


@register(jax.lax.dot_general_p)
def dot_general_cartnd(
    lhs: CartesianPosND, rhs: CartesianPosND, /, **kwargs: Any
) -> AbstractQuantity:
    """Dot product of two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q1 = cx.vecs.CartesianPosND.from_([1, 2, 3], "m")
    >>> q2 = cx.vecs.CartesianPosND.from_([4, 5, 6], "m")

    >>> jnp.dot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    """
    return qlax.dot_general(lhs.q, rhs.q, **kwargs)


# ------------------------------------------------


@register(jax.lax.mul_p)
def mul_vcnd(lhs: ArrayLike, rhs: CartesianPosND, /) -> CartesianPosND:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], "km"))
    >>> jnp.multiply(2, v).q
    Quantity['length'](Array([ 2,  4,  6,  8, 10], dtype=int32), unit='km')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, q=lhs * rhs.q)


@register(jax.lax.neg_p)
def neg_p_cartnd_pos(obj: CartesianPosND, /) -> CartesianPosND:
    """Negate the `coordinax.CartesianPosND`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 3D vector:

    >>> vec = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
    >>> (-vec).q
    Quantity['length'](Array([-1, -2, -3], dtype=int32), unit='km')

    """
    return jax.tree.map(qlax.neg, obj)


@register(jax.lax.sub_p)
def sub_cnd_pos(lhs: CartesianPosND, rhs: AbstractPos, /) -> CartesianPosND:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 3D vector:

    >>> q1 = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
    >>> q2 = cx.vecs.CartesianPosND(u.Quantity([2, 3, 4], "km"))
    >>> print(q1 - q2)
    <CartesianPosND (q[km])
        [[-1]
         [-1]
         [-1]]>

    """
    cart = cast(CartesianPosND, rhs.vconvert(CartesianPosND))
    return replace(lhs, q=lhs.q - cart.q)
