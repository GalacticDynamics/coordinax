"""Register Angle support for jax primitives."""
# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from typing import Any, TypeVar

from jax import lax
from jaxtyping import ArrayLike
from plum import convert
from quax import register

import unxt as u
from quaxed import lax as qlax

from .base import AbstractAngle

T = TypeVar("T")

one = u.unit("")
radian = u.unit("radian")


# TODO: can this be done with promotion/conversion instead?
@register(lax.cbrt_p)  # type: ignore[misc]
def _cbrt_p_a(x: AbstractAngle) -> u.Quantity:
    """Cube root of an angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(8, "rad")
    >>> jnp.cbrt(q)
    Quantity['rad1/3'](Array(2., dtype=float32, weak_type=True), unit='rad(1/3)')

    """
    return qlax.cbrt(convert(x, u.Quantity))


# ==============================================================================


@register(lax.cos_p)  # type: ignore[misc]
def _cos_p(x: AbstractAngle) -> u.Quantity:
    """Cosine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(0, "deg")
    >>> jnp.cos(q)
    Quantity['dimensionless'](Array(1., dtype=float32, ...), unit='')

    """
    return qlax.cos(convert(x, u.Quantity))


# ==============================================================================


@register(lax.dot_general_p)  # type: ignore[misc]
def _dot_general_aa(
    lhs: AbstractAngle, rhs: AbstractAngle, /, **kwargs: Any
) -> u.Quantity:
    """Dot product of two Angles.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q1 = Angle([1, 2, 3], "deg")
    >>> q2 = Angle([4, 5, 6], "deg")
    >>> jnp.vecdot(q1, q2)
    Quantity['solid angle'](Array(32, dtype=int32), unit='deg2')

    >>> q1 @ q2
    Quantity['solid angle'](Array(32, dtype=int32), unit='deg2')

    """
    return u.Quantity(
        lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs),
        unit=lhs.unit * rhs.unit,
    )


# ==============================================================================


@register(lax.integer_pow_p)  # type: ignore[misc]
def _integer_pow_p_a(x: AbstractAngle, *, y: Any) -> u.Quantity:
    """Integer power of an Angle.

    Examples
    --------
    >>> from coordinax.angle import Angle
    >>> q = Angle(2, "deg")

    >>> q ** 3
    Quantity['rad3'](Array(8, dtype=int32, weak_type=True), unit='deg3')

    """
    return qlax.integer_pow(convert(x, u.Quantity), y)


# ==============================================================================


@register(lax.pow_p)  # type: ignore[misc]
def _pow_p_a(x: AbstractAngle, y: ArrayLike) -> u.Quantity:
    """Power of an Angle by redispatching to Quantity.

    Examples
    --------
    >>> import math
    >>> from coordinax.angle import Angle

    >>> q1 = Angle(10.0, "deg")
    >>> y = 3.0
    >>> q1 ** y
    Quantity['rad3'](Array(1000., dtype=float32, ...), unit='deg3')

    """
    return qlax.pow(convert(x, u.Quantity), y)


# ==============================================================================


@register(lax.sin_p)  # type: ignore[misc]
def _sin_p(x: AbstractAngle) -> u.Quantity:
    """Sine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(90, "deg")
    >>> jnp.sin(q)
    Quantity['dimensionless'](Array(1., dtype=float32, ...), unit='')

    """
    return qlax.sin(convert(x, u.Quantity))


# ==============================================================================


@register(lax.sqrt_p)  # type: ignore[misc]
def _sqrt_p_a(x: AbstractAngle) -> u.Quantity:
    """Square root of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(9, "deg")
    >>> jnp.sqrt(q)
    Quantity['rad0.5'](Array(3., dtype=float32, ...), unit='deg(1/2)')

    """
    return qlax.sqrt(convert(x, u.Quantity))


# ==============================================================================


@register(lax.tan_p)  # type: ignore[misc]
def _tan_p_a(x: AbstractAngle) -> u.Quantity:
    """Tangent of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(45, "deg")
    >>> jnp.tan(q)
    Quantity['dimensionless'](Array(1., dtype=float32, ...), unit='')

    """
    return qlax.tan(convert(x, u.Quantity))
