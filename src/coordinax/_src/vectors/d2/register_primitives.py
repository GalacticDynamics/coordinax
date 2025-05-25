"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from dataclassish import replace

from .cartesian import CartesianAcc2D, CartesianPos2D, CartesianVel2D
from .polar import PolarPos
from coordinax._src.vectors.base_pos import AbstractPos

# -----------------------------------------------------


@register(jax.lax.add_p)
def add_p_cart2d_pos(lhs: CartesianPos2D, rhs: AbstractPos, /) -> CartesianPos2D:
    """Add a CartesianPos2D and position vector.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> polr = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))
    >>> print(cart + polr)
    <CartesianPos2D: (x, y) [km]
        [1. 5.]>

    >>> print(jnp.add(cart, polr))
    <CartesianPos2D: (x, y) [km]
        [1. 5.]>

    """
    cart = rhs.vconvert(CartesianPos2D)
    return jax.tree.map(jnp.add, lhs, cart)


@register(jax.lax.add_p)
def add_p_pp(lhs: CartesianVel2D, rhs: CartesianVel2D, /) -> CartesianVel2D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> print(v + v)
    <CartesianVel2D: (x, y) [km / s]
        [2 4]>

    >>> print(jnp.add(v, v))
    <CartesianVel2D: (x, y) [km / s]
        [2 4]>

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.add_p)
def add_p_aa(lhs: CartesianAcc2D, rhs: CartesianAcc2D, /) -> CartesianAcc2D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "km/s2")
    >>> print(v + v)
    <CartesianAcc2D: (x, y) [km / s2]
        [6 8]>

    >>> print(jnp.add(v, v))
    <CartesianAcc2D: (x, y) [km / s2]
        [6 8]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


# ------------------------------------------------
# Dot product
# TODO: see implementation in https://github.com/google/tree-math for how to do
# this more generally.


# TODO: this works for more Cartesian types
@register(jax.lax.dot_general_p)
def dot_general_p_cart2d(
    lhs: CartesianPos2D, rhs: CartesianPos2D, /, **kwargs: Any
) -> u.AbstractQuantity:
    """Dot product of two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q1 = cx.vecs.CartesianPos2D.from_([1, 2], "m")
    >>> q2 = cx.vecs.CartesianPos2D.from_([3, 4], "m")

    >>> jnp.dot(q1, q2)
    Quantity(Array(11, dtype=int32), unit='m2')

    """
    tree = jax.tree.map(jnp.multiply, lhs, rhs, is_leaf=u.quantity.is_any_quantity)
    return jax.tree.reduce(jnp.add, tree, is_leaf=u.quantity.is_any_quantity)


# ------------------------------------------------


@register(jax.lax.mul_p)
def mul_p_v_cart2d(lhs: ArrayLike, rhs: CartesianPos2D, /) -> CartesianPos2D:
    """Scale a cartesian 2D position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianPos2D.from_([3, 4], "m")
    >>> jnp.multiply(5, v).x
    Quantity(Array(15, dtype=int32), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y)


@register(jax.lax.mul_p)
def mul_p_vp(lhs: ArrayLike, rhs: CartesianVel2D, /) -> CartesianVel2D:
    """Scale a cartesian 2D velocity by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel2D.from_([3, 4], "m/s")
    >>> print(5 * v)
    <CartesianVel2D: (x, y) [m / s]
        [15 20]>

    >>> print(jnp.multiply(5, v))
    <CartesianVel2D: (x, y) [m / s]
        [15 20]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y)


@register(jax.lax.mul_p)
def mul_p_va(lhs: ArrayLike, rhs: CartesianAcc2D, /) -> CartesianAcc2D:
    """Scale a cartesian 2D acceleration by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "m/s2")
    >>> print(jnp.multiply(5, v))
    <CartesianAcc2D: (x, y) [m / s2]
        [15 20]>

    >>> print(5 * v)
    <CartesianAcc2D: (x, y) [m / s2]
        [15 20]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y)


@register(jax.lax.mul_p)
def mul_p_v_polar(lhs: ArrayLike, rhs: PolarPos, /) -> PolarPos:
    """Scale the polar position by a scalar.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import quaxed

    >>> v = cx.vecs.PolarPos(r=u.Quantity(1, "m"), phi=u.Quantity(90, "deg"))
    >>> print(v)
    <PolarPos: (r[m], phi[deg])
        [ 1 90]>

    >>> quaxed.numpy.linalg.vector_norm(v, axis=-1)
    BareQuantity(Array(1., dtype=float32), unit='m')

    >>> nv = quaxed.lax.mul(2, v)
    >>> print(nv)
    <PolarPos: (r[m], phi[deg])
        [ 2 90]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )
    # Scale the radial distance
    return replace(rhs, r=lhs * rhs.r)


# ------------------------------------------------


@register(jax.lax.neg_p)
def neg_p_cart2d_pos(obj: CartesianPos2D, /) -> CartesianPos2D:
    """Negate the `coordinax.vecs.CartesianPos2D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> (-q).x
    Quantity(Array(-1, dtype=int32), unit='km')

    """
    return jax.tree.map(qlax.neg, obj)


# ------------------------------------------------


@register(jax.lax.sub_p)
def sub_p_cart2d_pos2d(lhs: CartesianPos2D, rhs: AbstractPos, /) -> CartesianPos2D:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> polr = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))

    >>> print(cart - polr)
    <CartesianPos2D: (x, y) [km]
        [ 1. -1.]>

    """
    cart = rhs.vconvert(CartesianPos2D)
    return jax.tree.map(jnp.subtract, lhs, cart)
