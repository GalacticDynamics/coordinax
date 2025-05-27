"""Register primitives for 1D vector classes."""

__all__: list[str] = []

from typing import Any, cast

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from dataclassish import replace

from .cartesian import CartesianAcc1D, CartesianPos1D, CartesianVel1D
from coordinax._src.vectors.base_pos import AbstractPos

# ---------------------------------------------------------


@register(jax.lax.add_p)
def add_qq(lhs: CartesianPos1D, rhs: AbstractPos, /) -> CartesianPos1D:
    """Add a vector to a CartesianPos1D.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> r = cx.vecs.RadialPos.from_([1], "km")

    >>> print(jnp.add(q, r))
    <CartesianPos1D: (x) [km]
        [2]>

    >>> print(q + r)
    <CartesianPos1D: (x) [km]
        [2]>

    """
    rhs = cast(CartesianPos1D, rhs.vconvert(CartesianPos1D))
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.add_p)
def add_pp(lhs: CartesianVel1D, rhs: CartesianVel1D, /) -> CartesianVel1D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel1D.from_([1], "km/s")
    >>> vec = jnp.add(v, v)
    >>> print(vec)
    <CartesianVel1D: (x) [km / s]
        [2]>

    >>> print(v + v)
    <CartesianVel1D: (x) [km / s]
        [2]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.add_p)
def add_aa(lhs: CartesianAcc1D, rhs: CartesianAcc1D, /) -> CartesianAcc1D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc1D.from_([1], "km/s2")
    >>> vec = jnp.add(v, v)
    >>> print(vec)
    <CartesianAcc1D: (x) [km / s2]
        [2]>

    >>> print(v + v)
    <CartesianAcc1D: (x) [km / s2]
        [2]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


# ------------------------------------------------
# Dot product
# TODO: see implementation in https://github.com/google/tree-math for how to do
# this more generally.


@register(jax.lax.dot_general_p)
def dot_general_cart1d(
    lhs: CartesianPos1D, rhs: CartesianPos1D, /, **kwargs: Any
) -> u.AbstractQuantity:
    """Dot product of two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q1 = cx.vecs.CartesianPos1D.from_([1], "m")
    >>> q2 = cx.vecs.CartesianPos1D.from_([2], "m")

    >>> jnp.dot(q1, q2)
    Quantity(Array(2, dtype=int32), unit='m2')

    """
    return lhs.x * rhs.x


# ------------------------------------------------


@register(jax.lax.mul_p)
def mul_ac1(lhs: ArrayLike, rhs: CartesianPos1D, /) -> CartesianPos1D:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianPos1D.from_(1, "m")
    >>> jnp.multiply(2, v).x
    Quantity(Array(2, dtype=int32), unit='m')

    >>> (2 * v).x
    Quantity(Array(2, dtype=int32, ...), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x)


@register(jax.lax.mul_p)
def mul_vcart(lhs: ArrayLike, rhs: CartesianVel1D, /) -> CartesianVel1D:
    """Scale a velocity by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel1D.from_(1, "m/s")
    >>> vec = jnp.multiply(2, v)
    >>> print(vec)
    <CartesianVel1D: (x) [m / s]
        [2]>

    >>> print(2 * v)
    <CartesianVel1D: (x) [m / s]
        [2]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x)


@register(jax.lax.mul_p)
def mul_aq(lhs: ArrayLike, rhs: CartesianAcc1D, /) -> CartesianAcc1D:
    """Scale an acceleration by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc1D.from_(1, "m/s2")
    >>> vec = jnp.multiply(2, v)
    >>> print(vec)
    <CartesianAcc1D: (x) [m / s2]
        [2]>

    >>> print(2 * v)
    <CartesianAcc1D: (x) [m / s2]
        [2]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x)


# ------------------------------------------------


@register(jax.lax.neg_p)
def neg_p_cart1d_pos(obj: CartesianPos1D, /) -> CartesianPos1D:
    """Negate the `coordinax.CartesianPos1D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> (-q).x
    Quantity(Array(-1, dtype=int32), unit='km')

    """
    return jax.tree.map(qlax.neg, obj)


# ------------------------------------------------


@register(jax.lax.sub_p)
def sub_q1d_pos(self: CartesianPos1D, other: AbstractPos, /) -> CartesianPos1D:
    """Subtract two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> r = cx.vecs.RadialPos.from_([1], "km")

    >>> print(jnp.subtract(q, r))
    <CartesianPos1D: (x) [km]
        [0]>

    >>> print(q - r)
    <CartesianPos1D: (x) [km]
        [0]>

    """
    cart = other.vconvert(CartesianPos1D)
    return jax.tree.map(jnp.subtract, self, cart)


@register(jax.lax.sub_p)
def sub_a1_a1(self: CartesianAcc1D, other: CartesianAcc1D, /) -> CartesianAcc1D:
    """Subtract two 1-D cartesian accelerations.

    Examples
    --------
    >>> from quaxed import lax
    >>> import coordinax as cx

    >>> v1 = cx.vecs.CartesianAcc1D.from_(1, "m/s2")
    >>> v2 = cx.vecs.CartesianAcc1D.from_(2, "m/s2")
    >>> vec = lax.sub(v1, v2)
    >>> print(vec)
    <CartesianAcc1D: (x) [m / s2]
        [-1]>

    >>> print(v1 - v2)
    <CartesianAcc1D: (x) [m / s2]
        [-1]>

    """
    return jax.tree.map(jnp.subtract, self, other)
