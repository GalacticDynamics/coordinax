"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from collections.abc import Callable
from typing import TypeAlias, TypeVar

from jaxtyping import Shaped
from plum import conversion_method as _conversion_method, convert, dispatch

import quaxed.numpy as xp
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
    <CartesianPos1D (x[km])
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1], "km")))
    <CartesianPos1D (x[km])
        [1]>

    >>> print(cx.vecs.vector(u.Quantity(1, "km/s")))
    <CartesianVel1D (x[km / s])
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1], "km/s")))
    <CartesianVel1D (x[km / s])
        [1]>

    >>> print(cx.vecs.vector(u.Quantity(1, "km/s2")))
    <CartesianAcc1D (x[km / s2])
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1], "km/s2")))
    <CartesianAcc1D (x[km / s2])
        [1]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2], "km")))
    <CartesianPos2D (x[km], y[km])
        [1 2]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2], "km/s")))
    <CartesianVel2D (x[km / s], y[km / s])
        [1 2]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2], "km/s2")))
    <CartesianAcc2D (x[km / s2], y[km / s2])
        [1 2]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2, 3], "km")))
    <CartesianPos3D (x[km], y[km], z[km])
        [1 2 3]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2, 3], "km/s")))
    <CartesianVel3D (x[km / s], y[km / s], z[km / s])
        [1 2 3]>

    >>> print(cx.vecs.vector(u.Quantity([1, 2, 3], "km/s2")))
    <CartesianAcc3D (x[km / s2], y[km / s2], z[km / s2])
        [1 2 3]>

    >>> print(cx.vecs.vector(u.Quantity([0, 1, 2, 3], "km")))
    <CartesianPosND (q[km])
        [[0]
         [1]
         [2]
         [3]]>

    >>> print(cx.vecs.vector(u.Quantity([0, 1, 2, 3], "km/s")))
    <CartesianVelND (q[km / s])
        [[0]
         [1]
         [2]
         [3]]>

    >>> print(cx.vecs.vector(u.Quantity([0, 1, 2, 3], "km/s2")))
    <CartesianAccND (q[km / s2])
        [[0]
         [1]
         [2]
         [3]]>

    >>> try: print(cx.vecs.vector(u.Quantity([1], "Msun")))
    ... except ValueError as e: print(e)
    Cannot construct a Cartesian vector from Quantity['mass'](Array([1], dtype=int32), unit='solMass').

    """  # noqa: E501
    # TODO: use dispatch instead for these matches
    match (u.dimension_of(q), xp.atleast_1d(q).shape[-1]):
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
    return xp.stack(tuple(field_values(full_shaped(obj))), axis=-1)


# -------------------------------------------------------------------
# 1D

QConvertible1D: TypeAlias = CartesianVel1D | CartesianAcc1D | RadialVel | RadialAcc


@conversion_method(type_from=RadialAcc, type_to=BareQuantity)
@conversion_method(type_from=RadialVel, type_to=BareQuantity)
@conversion_method(type_from=CartesianAcc1D, type_to=BareQuantity)
@conversion_method(type_from=CartesianVel1D, type_to=BareQuantity)
def vec_diff1d_to_uncheckedq(
    obj: QConvertible1D, /
) -> Shaped[BareQuantity, "*batch 1"]:
    """1D Differentials -> `unxt.BareQuantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity
    >>> import coordinax as cx

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

    """
    return convert(_vec_diff_to_q(obj), BareQuantity)


@conversion_method(type_from=RadialAcc, type_to=u.Quantity)
@conversion_method(type_from=RadialVel, type_to=u.Quantity)
@conversion_method(type_from=CartesianAcc1D, type_to=u.Quantity)
@conversion_method(type_from=CartesianVel1D, type_to=u.Quantity)
def vec_diff1d_to_q(obj: QConvertible1D, /) -> Shaped[u.Quantity, "*batch 1"]:
    """1D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cart_vel = cx.vecs.CartesianVel1D.from_([1], "km/s")
    >>> convert(cart_vel, u.Quantity)
    Quantity['speed'](Array([1], dtype=int32), unit='km / s')

    >>> cart_acc = cx.vecs.CartesianAcc1D.from_([1], "km/s2")
    >>> convert(cart_acc, u.Quantity)
    Quantity['acceleration'](Array([1], dtype=int32), unit='km / s2')

    >>> rad_vel = cx.vecs.RadialVel.from_([1], "km/s")
    >>> convert(rad_vel, u.Quantity)
    Quantity['speed'](Array([1], dtype=int32), unit='km / s')

    >>> rad_acc = cx.vecs.RadialAcc.from_([1], "km/s2")
    >>> convert(rad_acc, u.Quantity)
    Quantity['acceleration'](Array([1], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), u.Quantity)


# -------------------------------------------------------------------
# 2D

QConvertible2D: TypeAlias = CartesianVel2D | CartesianAcc2D


@conversion_method(type_from=CartesianAcc2D, type_to=BareQuantity)
@conversion_method(type_from=CartesianVel2D, type_to=BareQuantity)
def vec_diff2d_to_uncheckedq(
    obj: QConvertible2D, /
) -> Shaped[BareQuantity, "*batch 2"]:
    """2D Differentials -> `unxt.BareQuantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity
    >>> import coordinax as cx

    >>> vel = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> convert(vel, BareQuantity)
    BareQuantity(Array([1, 2], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc2D.from_([1, 2], "km/s2")
    >>> convert(acc, BareQuantity)
    BareQuantity(Array([1, 2], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), BareQuantity)


@conversion_method(type_from=CartesianAcc2D, type_to=u.Quantity)
@conversion_method(type_from=CartesianVel2D, type_to=u.Quantity)
def vec_diff2d_to_q(obj: QConvertible2D, /) -> Shaped[u.Quantity, "*batch 2"]:
    """2D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vel = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> convert(vel, u.Quantity)
    Quantity['speed'](Array([1, 2], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc2D.from_([1, 2], "km/s2")
    >>> convert(acc, u.Quantity)
    Quantity['acceleration'](Array([1, 2], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), u.Quantity)


# -------------------------------------------------------------------
# 3D

QConvertible3D: TypeAlias = CartesianVel3D | CartesianAcc3D


@conversion_method(CartesianAcc3D, BareQuantity)
@conversion_method(CartesianVel3D, BareQuantity)
def vec_diff3d_to_uncheckedq(
    obj: QConvertible3D, /
) -> Shaped[BareQuantity, "*batch 3"]:
    """3D Differentials -> `unxt.BareQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity

    >>> vel = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(vel, BareQuantity)
    BareQuantity(Array([1, 2, 3], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(acc, BareQuantity)
    BareQuantity(Array([1, 2, 3], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), BareQuantity)


@conversion_method(CartesianAcc3D, u.Quantity)
@conversion_method(CartesianVel3D, u.Quantity)
def vec_diff3d_to_q(obj: QConvertible3D, /) -> Shaped[u.Quantity, "*batch 3"]:
    """3D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> import unxt as u

    >>> vel = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(vel, u.Quantity)
    Quantity['speed'](Array([1, 2, 3], dtype=int32), unit='km / s')

    >>> acc = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(acc, u.Quantity)
    Quantity['acceleration'](Array([1, 2, 3], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), u.Quantity)
