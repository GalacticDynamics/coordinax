"""Register primitives."""

__all__: list[str] = []


from dataclasses import replace
from typing import Any, cast

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u

from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D
from .generic import Cartesian3D
from .mathspherical import MathSphericalPos
from coordinax._src.vectors.base_pos import AbstractPos

# ------------------------------------------------


@register(jax.lax.add_p)
def add_cart3d_pos(lhs: CartesianPos3D, rhs: AbstractPos, /) -> CartesianPos3D:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> s = cx.SphericalPos(r=u.Quantity(1, "km"), theta=u.Quantity(90, "deg"),
    ...                     phi=u.Quantity(0, "deg"))
    >>> print(q + s)
    <CartesianPos3D: (x, y, z) [km]
        [2. 2. 3.]>

    """
    cart = rhs.vconvert(CartesianPos3D)
    return jax.tree.map(jnp.add, lhs, cart, is_leaf=u.quantity.is_any_quantity)


@register(jax.lax.add_p)
def add_pp(lhs: CartesianVel3D, rhs: CartesianVel3D, /) -> CartesianVel3D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> print(q + q)
    <CartesianVel3D: (x, y, z) [km / s]
        [2 4 6]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.add_p)
def add_aa(lhs: CartesianAcc3D, rhs: CartesianAcc3D, /) -> CartesianAcc3D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> print(q + q)
    <CartesianAcc3D: (x, y, z) [km / s2]
        [2 4 6]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


# ------------------------------------------------
# Dot product
# TODO: see implementation in https://github.com/google/tree-math for how to do
# this more generally.


@register(jax.lax.dot_general_p)
def dot_general_cart3d(
    lhs: CartesianPos3D, rhs: CartesianPos3D, /, **kwargs: Any
) -> u.AbstractQuantity:
    """Dot product of two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q1 = cx.vecs.CartesianPos3D.from_([1, 2, 3], "m")
    >>> q2 = cx.vecs.CartesianPos3D.from_([4, 5, 6], "m")

    >>> jnp.dot(q1, q2)
    Quantity(Array(32, dtype=int32), unit='m2')

    """
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z


# ------------------------------------------------


@register(jax.lax.mul_p)
def mul_p_vmsph(lhs: ArrayLike, rhs: MathSphericalPos, /) -> MathSphericalPos:
    """Scale the polar position by a scalar.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> v = cx.vecs.MathSphericalPos(r=u.Quantity(3, "km"),
    ...                              theta=u.Quantity(90, "deg"),
    ...                              phi=u.Quantity(0, "deg"))

    >>> jnp.linalg.vector_norm(v, axis=-1)
    BareQuantity(Array(3., dtype=float32), unit='km')

    >>> nv = jnp.multiply(2, v)
    >>> print(nv)
    <MathSphericalPos: (r[km], theta[deg], phi[deg])
        [ 6 90  0]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )
    # Scale the radial distance
    return replace(rhs, r=cast(u.AbstractQuantity, lhs * rhs.r))


@register(jax.lax.mul_p)
def mul_p_arraylike_cart3d(lhs: ArrayLike, rhs: CartesianPos3D, /) -> CartesianPos3D:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.CartesianPos3D.from_([1, 2, 3], "km")

    >>> print(2 * v)
    <CartesianPos3D: (x, y, z) [km]
        [2 4 6]>

    >>> print(jnp.multiply(2, v))
    <CartesianPos3D: (x, y, z) [km]
        [2 4 6]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y, z=lhs * rhs.z)


# ------------------------------------------------


@register(jax.lax.neg_p)
def neg_p_cart3d_pos(obj: CartesianPos3D, /) -> CartesianPos3D:
    """Negate the `coordinax.CartesianPos3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(-q)
    <CartesianPos3D: (x, y, z) [km]
        [-1 -2 -3]>

    """
    return jax.tree.map(qlax.neg, obj)


@register(jax.lax.neg_p)
def neg_p_genericcart3d(obj: Cartesian3D, /) -> Cartesian3D:
    """Negate the `coordinax.vecs.Cartesian3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.vecs.Cartesian3D.from_([1, 2, 3], "km")
    >>> print(-q)
    <Cartesian3D: (x, y, z) [km]
    [-1 -2 -3]>

    """
    return jax.tree.map(jnp.negative, obj)


# ------------------------------------------------


@register(jax.lax.sub_p)
def sub_p_cart3d_pos(lhs: CartesianPos3D, rhs: AbstractPos, /) -> CartesianPos3D:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> s = cx.SphericalPos(r=u.Quantity(1, "km"), theta=u.Quantity(90, "deg"),
    ...                     phi=u.Quantity(0, "deg"))
    >>> print(q - s)
    <CartesianPos3D: (x, y, z) [km]
        [0. 2. 3.]>

    """
    cart = rhs.vconvert(CartesianPos3D)
    return jax.tree.map(jnp.subtract, lhs, cart)


@register(jax.lax.sub_p)
def sub_p_v3_v3(lhs: CartesianVel3D, other: CartesianVel3D, /) -> CartesianVel3D:
    """Subtract two differentials.

    Examples
    --------
    >>> from coordinax import CartesianPos3D, CartesianVel3D
    >>> q = CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> print(q - q)
    <CartesianVel3D: (x, y, z) [km / s]
        [0 0 0]>

    """
    return jax.tree.map(jnp.subtract, lhs, other)


@register(jax.lax.sub_p)
def sub_p_a3_a3(lhs: CartesianAcc3D, rhs: CartesianAcc3D, /) -> CartesianAcc3D:
    """Subtract two accelerations.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> print(q - q)
    <CartesianAcc3D: (x, y, z) [km / s2]
        [0 0 0]>

    """
    return jax.tree.map(jnp.subtract, lhs, rhs)
