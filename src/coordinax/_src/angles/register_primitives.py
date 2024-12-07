"""Register Angle support for jax primitives."""
# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from collections.abc import Callable
from typing import Any, TypeVar

from jax import lax
from jax.core import Primitive
from jaxtyping import ArrayLike
from quax import register as register_

import unxt as u
from quaxed import lax as qlax

from .base import AbstractAngle

T = TypeVar("T")

one = u.unit("")
radian = u.unit("radian")


def register(primitive: Primitive, **kwargs: Any) -> Callable[[T], T]:
    """`quax.register`, but makes mypy happy."""
    return register_(primitive, **kwargs)


################################################################################
# Registering Primitives


# TODO: can this be done with promotion/conversion instead?
@register(lax.cbrt_p)
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
    return u.Quantity(lax.cbrt(x.value), unit=x.unit ** (1 / 3))


# ==============================================================================


@register(lax.cos_p)
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
    return u.Quantity(qlax.cos(u.ustrip(radian, x)), unit=one)


# ==============================================================================


@register(lax.dot_general_p)
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


@register(lax.integer_pow_p)
def _integer_pow_p_a(x: AbstractAngle, *, y: Any) -> u.Quantity:
    """Integer power of an Angle.

    Examples
    --------
    >>> from coordinax.angle import Angle
    >>> q = Angle(2, "deg")

    >>> q ** 3
    Quantity['rad3'](Array(8, dtype=int32, weak_type=True), unit='deg3')

    """
    return u.Quantity(value=lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


@register(lax.pow_p)
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
    return u.Quantity(x.value, x.unit) ** y  # TODO: better call to power


# ==============================================================================


@register(lax.sin_p)
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
    return u.Quantity(qlax.sin(u.ustrip(radian, x)), unit=one)


# ==============================================================================


@register(lax.sqrt_p)
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
    # Promote to something that supports sqrt units.
    return u.Quantity(lax.sqrt(x.value), unit=x.unit ** (1 / 2))
