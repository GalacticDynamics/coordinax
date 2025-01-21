"""Register Angle support for jax primitives."""
# pylint: disable=import-error

__all__: list[str] = []

from typing import Any, TypeVar

from jax import lax
from jaxtyping import ArrayLike
from plum import convert
from quax import register

import unxt as u
from quaxed import lax as qlax
from unxt.quantity import UncheckedQuantity as FastQ

from .base import AbstractAngle

T = TypeVar("T")

one = u.unit("")
radian = u.unit("radian")


# TODO: can this be done with promotion/conversion/default rule instead?
@register(lax.cbrt_p)
def cbrt_p_a(x: AbstractAngle) -> FastQ:
    """Cube root of an angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(8, "rad")
    >>> jnp.cbrt(q)
    UncheckedQuantity(Array(2., dtype=float32, weak_type=True), unit='rad(1/3)')

    """
    return qlax.cbrt(convert(x, FastQ))


# ==============================================================================


@register(lax.cos_p)
def cos_p(x: AbstractAngle) -> FastQ:
    """Cosine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(0, "deg")
    >>> jnp.cos(q)
    UncheckedQuantity(Array(1., dtype=float32, weak_type=True), unit='')

    """
    return qlax.cos(convert(x, FastQ))


# ==============================================================================


@register(lax.dot_general_p)
def dot_general_aa(lhs: AbstractAngle, rhs: AbstractAngle, /, **kwargs: Any) -> FastQ:
    """Dot product of two Angles.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q1 = Angle([1, 2, 3], "deg")
    >>> q2 = Angle([4, 5, 6], "deg")
    >>> jnp.vecdot(q1, q2)
    UncheckedQuantity(Array(32, dtype=int32), unit='deg2')

    >>> q1 @ q2
    UncheckedQuantity(Array(32, dtype=int32), unit='deg2')

    """
    value = lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs)
    return FastQ(value, unit=lhs.unit * rhs.unit)


# ==============================================================================


@register(lax.integer_pow_p)
def integer_pow_p_a(x: AbstractAngle, *, y: Any) -> FastQ:
    """Integer power of an Angle.

    Examples
    --------
    >>> from coordinax.angle import Angle
    >>> q = Angle(2, "deg")

    >>> q ** 3
    UncheckedQuantity(Array(8, dtype=int32, weak_type=True), unit='deg3')

    """
    return qlax.integer_pow(convert(x, FastQ), y)


# ==============================================================================


@register(lax.pow_p)
def pow_p_a(x: AbstractAngle, y: ArrayLike) -> FastQ:
    """Power of an Angle by redispatching to Quantity.

    Examples
    --------
    >>> import math
    >>> from coordinax.angle import Angle

    >>> q1 = Angle(10.0, "deg")
    >>> y = 3.0
    >>> q1 ** y
    UncheckedQuantity(Array(1000., dtype=float32, weak_type=True), unit='deg3')

    """
    return qlax.pow(convert(x, FastQ), y)


# ==============================================================================


@register(lax.sin_p)
def sin_p(x: AbstractAngle) -> FastQ:
    """Sine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(90, "deg")
    >>> jnp.sin(q)
    UncheckedQuantity(Array(1., dtype=float32, weak_type=True), unit='')

    """
    return qlax.sin(convert(x, FastQ))


# ==============================================================================


@register(lax.sqrt_p)
def sqrt_p_a(x: AbstractAngle) -> FastQ:
    """Square root of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(9, "deg")
    >>> jnp.sqrt(q)
    UncheckedQuantity(Array(3., dtype=float32, weak_type=True), unit='deg(1/2)')

    """
    return qlax.sqrt(convert(x, FastQ))


# ==============================================================================


@register(lax.tan_p)
def tan_p_a(x: AbstractAngle) -> FastQ:
    """Tangent of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(45, "deg")
    >>> jnp.tan(q)
    UncheckedQuantity(Array(1., dtype=float32, weak_type=True), unit='')

    """
    return qlax.tan(convert(x, FastQ))
