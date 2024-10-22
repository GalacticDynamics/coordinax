"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.numpy as xp
from dataclassish import field_values
from unxt import AbstractQuantity, Quantity, UncheckedQuantity

from coordinax._src.base.base import AbstractVector
from coordinax._src.d1.cartesian import CartesianAcc1D, CartesianPos1D, CartesianVel1D
from coordinax._src.d1.radial import RadialAcc, RadialVel
from coordinax._src.d2.cartesian import CartesianAcc2D, CartesianVel2D
from coordinax._src.d3.cartesian import CartesianAcc3D, CartesianVel3D
from coordinax._src.operators.base import AbstractOperator, op_call_dispatch
from coordinax._src.typing import TimeBatchOrScalar
from coordinax._src.utils import full_shaped

#####################################################################
# Convert to Quantity


def _vec_diff_to_q(obj: AbstractVector, /) -> AbstractQuantity:
    """`coordinax.AbstractVector` -> `unxt.AbstractQuantity`."""
    return xp.stack(tuple(field_values(full_shaped(obj))), axis=-1)


# -------------------------------------------------------------------
# 1D

QConvertible1D: TypeAlias = CartesianVel1D | CartesianAcc1D | RadialVel | RadialAcc


@conversion_method(type_from=RadialAcc, type_to=UncheckedQuantity)
@conversion_method(type_from=RadialVel, type_to=UncheckedQuantity)
@conversion_method(type_from=CartesianAcc1D, type_to=UncheckedQuantity)
@conversion_method(type_from=CartesianVel1D, type_to=UncheckedQuantity)
def vec_diff1d_to_uncheckedq(
    obj: QConvertible1D, /
) -> Shaped[UncheckedQuantity, "*batch 1"]:
    """1D Differentials -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import UncheckedQuantity
    >>> import coordinax as cx

    >>> cart_vel = cx.CartesianVel1D.from_([1], "km/s")
    >>> convert(cart_vel, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s')

    >>> cart_acc = cx.CartesianAcc1D.from_([1], "km/s2")
    >>> convert(cart_acc, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s2')

    >>> rad_vel = cx.RadialVel.from_([1], "km/s")
    >>> convert(rad_vel, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s')

    >>> rad_acc = cx.RadialAcc.from_([1], "km/s2")
    >>> convert(rad_acc, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), UncheckedQuantity)


@conversion_method(type_from=RadialAcc, type_to=Quantity)
@conversion_method(type_from=RadialVel, type_to=Quantity)
@conversion_method(type_from=CartesianAcc1D, type_to=Quantity)
@conversion_method(type_from=CartesianVel1D, type_to=Quantity)
def vec_diff1d_to_uncheckedq(obj: QConvertible1D, /) -> Shaped[Quantity, "*batch 1"]:
    """1D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cart_vel = cx.CartesianVel1D.from_([1], "km/s")
    >>> convert(cart_vel, Quantity)
    Quantity['speed'](Array([1], dtype=int32), unit='km / s')

    >>> cart_acc = cx.CartesianAcc1D.from_([1], "km/s2")
    >>> convert(cart_acc, Quantity)
    Quantity['acceleration'](Array([1], dtype=int32), unit='km / s2')

    >>> rad_vel = cx.RadialVel.from_([1], "km/s")
    >>> convert(rad_vel, Quantity)
    Quantity['speed'](Array([1], dtype=int32), unit='km / s')

    >>> rad_acc = cx.RadialAcc.from_([1], "km/s2")
    >>> convert(rad_acc, Quantity)
    Quantity['acceleration'](Array([1], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), Quantity)


# -------------------------------------------------------------------
# 2D

QConvertible2D: TypeAlias = CartesianVel2D | CartesianAcc2D


@conversion_method(type_from=CartesianAcc2D, type_to=UncheckedQuantity)
@conversion_method(type_from=CartesianVel2D, type_to=UncheckedQuantity)
def vec_diff_to_q(obj: QConvertible2D, /) -> Shaped[UncheckedQuantity, "*batch 2"]:
    """2D Differentials -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import UncheckedQuantity
    >>> import coordinax as cx

    >>> vel = cx.CartesianVel2D.from_([1, 2], "km/s")
    >>> convert(vel, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcc2D.from_([1, 2], "km/s2")
    >>> convert(acc, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), UncheckedQuantity)


@conversion_method(type_from=CartesianAcc2D, type_to=Quantity)
@conversion_method(type_from=CartesianVel2D, type_to=Quantity)
def vec_diff_to_q(obj: QConvertible2D, /) -> Shaped[Quantity, "*batch 2"]:
    """2D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vel = cx.CartesianVel2D.from_([1, 2], "km/s")
    >>> convert(vel, Quantity)
    Quantity['speed'](Array([1., 2.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcc2D.from_([1, 2], "km/s2")
    >>> convert(acc, Quantity)
    Quantity['acceleration'](Array([1., 2.], dtype=float32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), Quantity)


# -------------------------------------------------------------------
# 3D

QConvertible3D: TypeAlias = CartesianVel3D | CartesianAcc3D


@conversion_method(CartesianAcc3D, UncheckedQuantity)
@conversion_method(CartesianVel3D, UncheckedQuantity)
def vec_diff_to_q(obj: CartesianVel3D, /) -> Shaped[UncheckedQuantity, "*batch 3"]:
    """3D Differentials -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import UncheckedQuantity

    >>> vel = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(vel, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(acc, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), UncheckedQuantity)


@conversion_method(CartesianAcc3D, Quantity)
@conversion_method(CartesianVel3D, Quantity)
def vec_diff_to_q(obj: CartesianVel3D, /) -> Shaped[Quantity, "*batch 3"]:
    """3D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> vel = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(vel, Quantity)
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(acc, Quantity)
    Quantity['acceleration'](Array([1., 2., 3.], dtype=float32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), Quantity)


#####################################################################
# Operators


Q1: TypeAlias = Shaped[Quantity["length"], "*#batch 1"]


@op_call_dispatch
def call(self: AbstractOperator, x: Q1, /) -> Q1:
    """Dispatch to the operator's `__call__` method."""
    return self(CartesianPos1D.from_(x))


@op_call_dispatch
def call(
    self: AbstractOperator, x: Q1, t: TimeBatchOrScalar, /
) -> tuple[Q1, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method."""
    vec, t = self(CartesianPos1D.from_(x), t)
    return convert(vec, Quantity), t
