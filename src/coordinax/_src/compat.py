"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.numpy as xp
from dataclassish import field_values
from unxt import AbstractQuantity, Quantity, UncheckedQuantity

from coordinax._src.base.base import AbstractVector
from coordinax._src.d1.cartesian import (
    CartesianAcceleration1D,
    CartesianPosition1D,
    CartesianVelocity1D,
)
from coordinax._src.d1.radial import RadialAcceleration, RadialVelocity
from coordinax._src.d2.cartesian import CartesianAcceleration2D, CartesianVelocity2D
from coordinax._src.d3.cartesian import CartesianAcceleration3D, CartesianVelocity3D
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

QConvertible1D: TypeAlias = (
    CartesianVelocity1D | CartesianAcceleration1D | RadialVelocity | RadialAcceleration
)


@conversion_method(type_from=RadialAcceleration, type_to=UncheckedQuantity)
@conversion_method(type_from=RadialVelocity, type_to=UncheckedQuantity)
@conversion_method(type_from=CartesianAcceleration1D, type_to=UncheckedQuantity)
@conversion_method(type_from=CartesianVelocity1D, type_to=UncheckedQuantity)
def vec_diff1d_to_uncheckedq(
    obj: QConvertible1D, /
) -> Shaped[UncheckedQuantity, "*batch 1"]:
    """1D Differentials -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import UncheckedQuantity
    >>> import coordinax as cx

    >>> cart_vel = cx.CartesianVelocity1D.constructor([1], "km/s")
    >>> convert(cart_vel, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s')

    >>> cart_acc = cx.CartesianAcceleration1D.constructor([1], "km/s2")
    >>> convert(cart_acc, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s2')

    >>> rad_vel = cx.RadialVelocity.constructor([1], "km/s")
    >>> convert(rad_vel, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s')

    >>> rad_acc = cx.RadialAcceleration.constructor([1], "km/s2")
    >>> convert(rad_acc, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), UncheckedQuantity)


@conversion_method(type_from=RadialAcceleration, type_to=Quantity)
@conversion_method(type_from=RadialVelocity, type_to=Quantity)
@conversion_method(type_from=CartesianAcceleration1D, type_to=Quantity)
@conversion_method(type_from=CartesianVelocity1D, type_to=Quantity)
def vec_diff1d_to_uncheckedq(obj: QConvertible1D, /) -> Shaped[Quantity, "*batch 1"]:
    """1D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cart_vel = cx.CartesianVelocity1D.constructor([1], "km/s")
    >>> convert(cart_vel, Quantity)
    Quantity['speed'](Array([1], dtype=int32), unit='km / s')

    >>> cart_acc = cx.CartesianAcceleration1D.constructor([1], "km/s2")
    >>> convert(cart_acc, Quantity)
    Quantity['acceleration'](Array([1], dtype=int32), unit='km / s2')

    >>> rad_vel = cx.RadialVelocity.constructor([1], "km/s")
    >>> convert(rad_vel, Quantity)
    Quantity['speed'](Array([1], dtype=int32), unit='km / s')

    >>> rad_acc = cx.RadialAcceleration.constructor([1], "km/s2")
    >>> convert(rad_acc, Quantity)
    Quantity['acceleration'](Array([1], dtype=int32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), Quantity)


# -------------------------------------------------------------------
# 2D

QConvertible2D: TypeAlias = CartesianVelocity2D | CartesianAcceleration2D


@conversion_method(type_from=CartesianAcceleration2D, type_to=UncheckedQuantity)
@conversion_method(type_from=CartesianVelocity2D, type_to=UncheckedQuantity)
def vec_diff_to_q(obj: QConvertible2D, /) -> Shaped[UncheckedQuantity, "*batch 2"]:
    """2D Differentials -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import UncheckedQuantity
    >>> import coordinax as cx

    >>> vel = cx.CartesianVelocity2D.constructor([1, 2], "km/s")
    >>> convert(vel, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcceleration2D.constructor([1, 2], "km/s2")
    >>> convert(acc, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), UncheckedQuantity)


@conversion_method(type_from=CartesianAcceleration2D, type_to=Quantity)
@conversion_method(type_from=CartesianVelocity2D, type_to=Quantity)
def vec_diff_to_q(obj: QConvertible2D, /) -> Shaped[Quantity, "*batch 2"]:
    """2D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vel = cx.CartesianVelocity2D.constructor([1, 2], "km/s")
    >>> convert(vel, Quantity)
    Quantity['speed'](Array([1., 2.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcceleration2D.constructor([1, 2], "km/s2")
    >>> convert(acc, Quantity)
    Quantity['acceleration'](Array([1., 2.], dtype=float32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), Quantity)


# -------------------------------------------------------------------
# 3D

QConvertible3D: TypeAlias = CartesianVelocity3D | CartesianAcceleration3D


@conversion_method(CartesianAcceleration3D, UncheckedQuantity)
@conversion_method(CartesianVelocity3D, UncheckedQuantity)
def vec_diff_to_q(obj: CartesianVelocity3D, /) -> Shaped[UncheckedQuantity, "*batch 3"]:
    """3D Differentials -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import UncheckedQuantity

    >>> vel = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
    >>> convert(vel, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcceleration3D.constructor([1, 2, 3], "km/s2")
    >>> convert(acc, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km / s2')

    """
    return convert(_vec_diff_to_q(obj), UncheckedQuantity)


@conversion_method(CartesianAcceleration3D, Quantity)
@conversion_method(CartesianVelocity3D, Quantity)
def vec_diff_to_q(obj: CartesianVelocity3D, /) -> Shaped[Quantity, "*batch 3"]:
    """3D Differentials -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> vel = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
    >>> convert(vel, Quantity)
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')

    >>> acc = cx.CartesianAcceleration3D.constructor([1, 2, 3], "km/s2")
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
    return self(CartesianPosition1D.constructor(x))


@op_call_dispatch
def call(
    self: AbstractOperator, x: Q1, t: TimeBatchOrScalar, /
) -> tuple[Q1, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method."""
    vec, t = self(CartesianPosition1D.constructor(x), t)
    return convert(vec, Quantity), t
