"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from collections.abc import Callable
from typing import TypeVar, Union

import equinox as eqx
from jaxtyping import Shaped
from jaxtyping._array_types import _MetaAbstractArray
from plum import conversion_method as _conversion_method, convert, dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_values
from unxt.quantity import BareQuantity

from .api import vector
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.d1 import (
    CartesianAcc1D,
    CartesianPos1D,
    CartesianVel1D,
    RadialAcc,
    RadialVel,
)
from coordinax._src.vectors.d2 import CartesianAcc2D, CartesianPos2D, CartesianVel2D
from coordinax._src.vectors.d3 import CartesianAcc3D, CartesianPos3D, CartesianVel3D
from coordinax._src.vectors.dn import CartesianAccND, CartesianPosND, CartesianVelND
from coordinax._src.vectors.utils import full_shaped

T = TypeVar("T")


def conversion_method(type_from: type, type_to: type) -> Callable[[T], T]:
    """Typed version of conversion method."""
    return _conversion_method(type_from, type_to)  # type: ignore[return-value]


#####################################################################
# TODO: move to unxt


@dispatch(precedence=1)
def dimension_of(obj: _MetaAbstractArray, /) -> u.dims.AbstractDimension:
    """Get the dimension of a jaxtyping-annotated object."""
    return u.dimension_of(obj.array_type)


@dispatch
def dimension_of(obj: Union, /) -> u.dims.AbstractDimension:
    # Call dimension_of on all the __args__ of the Union and make sure they are
    # all the same
    dims = {dimension_of(arg) for arg in obj.__args__}
    dims = eqx.error_if(dims, len(dims) != 1, "Cannot mix dimensions in a Union.")
    return dims.pop()


#####################################################################
# Construct from Quantity


class Dim:
    """Dimension enumeration."""

    LENGTH = u.dimension("length")
    SPEED = u.dimension("speed")
    ACCELERATION = u.dimension("acceleration")


@dispatch
def vector(q: u.AbstractQuantity, /) -> AbstractVector:  # noqa: C901
    """Construct a vector from a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> print(cx.vecs.vector(u.Quantity(1, "km")))
    <CartesianPos1D: (x) [km]
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1], "km")))
    <CartesianPos1D: (x) [km]
        [1]>

    >>> print(cx.vecs.vector(u.Quantity(1, "km/s")))
    <CartesianVel1D: (x) [km / s]
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1], "km/s")))
    <CartesianVel1D: (x) [km / s]
        [1]>

    >>> print(cx.vecs.vector(u.Quantity(1, "km/s2")))
    <CartesianAcc1D: (x) [km / s2]
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1], "km/s2")))
    <CartesianAcc1D: (x) [km / s2]
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2], "km")))
    <CartesianPos2D: (x, y) [km]
        [1 2]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2], "km/s")))
    <CartesianVel2D: (x, y) [km / s]
        [1 2]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2], "km/s2")))
    <CartesianAcc2D: (x, y) [km / s2]
        [1 2]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2, 3], "km")))
    <CartesianPos3D: (x, y, z) [km]
        [1 2 3]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2, 3], "km/s")))
    <CartesianVel3D: (x, y, z) [km / s]
        [1 2 3]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2, 3], "km/s2")))
    <CartesianAcc3D: (x, y, z) [km / s2]
        [1 2 3]>

    >>> print(cx.vecs.vector(u.Quantity([0, 1, 2, 3], "km")))
    <CartesianPosND: (q) [km]
        [[0]
         [1]
         [2]
         [3]]>

    >>> print(cx.vecs.vector(u.Quantity([0, 1, 2, 3], "km/s")))
    <CartesianVelND: (q) [km / s]
        [[0]
         [1]
         [2]
         [3]]>

    >>> print(cx.vecs.vector(u.Quantity([0, 1, 2, 3], "km/s2")))
    <CartesianAccND: (q) [km / s2]
        [[0]
         [1]
         [2]
         [3]]>

    >>> try: print(cx.vecs.vector(u.Quantity([1], "Msun")))
    ... except ValueError as e: print(e)
    Cannot construct a Cartesian vector from Quantity['mass']([1], unit='solMass').

    """
    # TODO: use dispatch instead for these matches
    match (u.dimension_of(q), jnp.atleast_1d(q).shape[-1]):
        case (Dim.LENGTH, 0) | (Dim.LENGTH, 1):
            return vector(CartesianPos1D, q)
        case (Dim.SPEED, 0) | (Dim.SPEED, 1):
            return vector(CartesianVel1D, q)
        case (Dim.ACCELERATION, 0) | (Dim.ACCELERATION, 1):
            return vector(CartesianAcc1D, q)
        case (Dim.LENGTH, 2):
            return vector(CartesianPos2D, q)
        case (Dim.SPEED, 2):
            return vector(CartesianVel2D, q)
        case (Dim.ACCELERATION, 2):
            return vector(CartesianAcc2D, q)
        case (Dim.LENGTH, 3):
            return vector(CartesianPos3D, q)
        case (Dim.SPEED, 3):
            return vector(CartesianVel3D, q)
        case (Dim.ACCELERATION, 3):
            return vector(CartesianAcc3D, q)
        case (Dim.LENGTH, _):
            return vector(CartesianPosND, q)
        case (Dim.SPEED, _):
            return vector(CartesianVelND, q)
        case (Dim.ACCELERATION, _):
            return vector(CartesianAccND, q)
        case _:
            msg = f"Cannot construct a Cartesian vector from {q}."
            raise ValueError(msg)


#####################################################################
# Convert to Quantity


def _vec_diff_to_q(obj: AbstractVector, /) -> u.AbstractQuantity:
    """`coordinax.AbstractVector` -> `unxt.u.AbstractQuantity`."""
    return jnp.stack(tuple(field_values(full_shaped(obj))), axis=-1)


@conversion_method(type_from=RadialAcc, type_to=BareQuantity)
@conversion_method(type_from=RadialVel, type_to=BareQuantity)
@conversion_method(type_from=CartesianAcc1D, type_to=BareQuantity)
@conversion_method(type_from=CartesianVel1D, type_to=BareQuantity)
@conversion_method(type_from=CartesianAcc2D, type_to=BareQuantity)
@conversion_method(type_from=CartesianVel2D, type_to=BareQuantity)
@conversion_method(type_from=CartesianAcc3D, type_to=BareQuantity)
@conversion_method(type_from=CartesianVel3D, type_to=BareQuantity)
def vec_diff_to_uncheckedq(obj: AbstractVector, /) -> Shaped[BareQuantity, "*batch N"]:
    """Differentials -> `unxt.BareQuantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity
    >>> import coordinax as cx

    ## 1D

    >>> cart_vel = cx.vecs.CartesianVel1D.from_([1], "km/s")
    >>> convert(cart_vel, BareQuantity)
    BareQuantity(Array([1], dtype=int32), unit='km / s')

    >>> cart_acc = cx.vecs.CartesianAcc1D.from_([1], "km/s2")
    >>> convert(cart_acc, BareQuantity)
    BareQuantity(Array([1], dtype=int32), unit='km / s2')

    >>> rad_vel = cx.vecs.RadialVel.from_([1], "km/s")
    >>> convert(rad_vel, BareQuantity)
    BareQuantity(Array([1], dtype=int32), unit='km / s')

    >>> rad_acc = cx.vecs.RadialAcc.from_([1], "km/s2")
    >>> convert(rad_acc, BareQuantity)
    BareQuantity(Array([1], dtype=int32), unit='km / s2')

    ## 2D

    >>> vel = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> convert(vel, BareQuantity)
    BareQuantity(Array([1, 2], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc2D.from_([1, 2], "km/s2")
    >>> convert(acc, BareQuantity)
    BareQuantity(Array([1, 2], dtype=int32), unit='km / s2')

    # 3D

    >>> vel = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(vel, BareQuantity)
    BareQuantity(Array([1, 2, 3], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(acc, BareQuantity)
    BareQuantity(Array([1, 2, 3], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), BareQuantity)


@conversion_method(type_from=RadialAcc, type_to=u.Quantity)
@conversion_method(type_from=RadialVel, type_to=u.Quantity)
@conversion_method(type_from=CartesianAcc1D, type_to=u.Quantity)
@conversion_method(type_from=CartesianVel1D, type_to=u.Quantity)
@conversion_method(type_from=CartesianAcc2D, type_to=u.Quantity)
@conversion_method(type_from=CartesianVel2D, type_to=u.Quantity)
@conversion_method(type_from=CartesianAcc3D, type_to=u.Quantity)
@conversion_method(type_from=CartesianVel3D, type_to=u.Quantity)
def vec_diff_to_q(obj: AbstractVector, /) -> Shaped[u.Quantity, "*batch N"]:
    """1D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    ## 1D

    >>> cart_vel = cx.vecs.CartesianVel1D.from_([1], "km/s")
    >>> convert(cart_vel, u.Quantity)
    Quantity(Array([1], dtype=int32), unit='km / s')

    >>> cart_acc = cx.vecs.CartesianAcc1D.from_([1], "km/s2")
    >>> convert(cart_acc, u.Quantity)
    Quantity(Array([1], dtype=int32), unit='km / s2')

    >>> rad_vel = cx.vecs.RadialVel.from_([1], "km/s")
    >>> convert(rad_vel, u.Quantity)
    Quantity(Array([1], dtype=int32), unit='km / s')

    >>> rad_acc = cx.vecs.RadialAcc.from_([1], "km/s2")
    >>> convert(rad_acc, u.Quantity)
    Quantity(Array([1], dtype=int32), unit='km / s2')

    ## 2D

    >>> vel = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> convert(vel, u.Quantity)
    Quantity(Array([1, 2], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc2D.from_([1, 2], "km/s2")
    >>> convert(acc, u.Quantity)
    Quantity(Array([1, 2], dtype=int32), unit='km / s2')

    # 3D

    >>> vel = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(vel, u.Quantity)
    Quantity(Array([1, 2, 3], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(acc, u.Quantity)
    Quantity(Array([1, 2, 3], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), u.Quantity)
