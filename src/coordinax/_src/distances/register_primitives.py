"""Register Quantity support for jax primitives."""
# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from collections.abc import Callable
from typing import Any, TypeVar

from jax import lax
from jax.core import Primitive
from jaxtyping import ArrayLike
from quax import register as register_

import unxt as u
from unxt.quantity import AbstractQuantity

from .base import AbstractDistance

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
def _cbrt_p_d(x: AbstractDistance) -> u.Quantity:
    """Cube root of a distance.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.distance import Distance
    >>> d = Distance(8, "m")
    >>> jnp.cbrt(d)
    Quantity['m1/3'](Array(2., dtype=float32, ...), unit='m(1/3)')

    """
    return u.Quantity(lax.cbrt(x.value), unit=x.unit ** (1 / 3))


# ==============================================================================


@register(lax.dot_general_p)
def _dot_general_dd(
    lhs: AbstractDistance, rhs: AbstractDistance, /, **kwargs: Any
) -> u.Quantity:
    """Dot product of two Distances.

    Examples
    --------
    This is a dot product of two Distances.

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.distance import Distance

    >>> q1 = Distance([1, 2, 3], "m")
    >>> q2 = Distance([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    Quantity['area'](Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    Quantity['area'](Array(32, dtype=int32), unit='m2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = jnp.asarray([[0, -1,  0], [1,  0,  0], [0,  0,  1]])
    >>> q = u.Quantity([1, 0, 0], "m")
    >>> Rz @ q
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    This uses `matmul` for quantities.

    >>> jnp.linalg.matmul(Rz, q)
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='m')

    """
    return u.Quantity(
        lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs),
        unit=lhs.unit * rhs.unit,
    )


# ==============================================================================


@register(lax.integer_pow_p)
def _integer_pow_p_d(x: AbstractDistance, *, y: Any) -> u.Quantity:
    """Integer power of a Distance.

    Examples
    --------
    >>> from coordinax.distance import Distance
    >>> q = Distance(2, "m")
    >>> q ** 3
    Quantity['volume'](Array(8, dtype=int32, ...), unit='m3')

    """
    return u.Quantity(value=lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


@register(lax.pow_p)
def _pow_p_d(x: AbstractDistance, y: ArrayLike) -> u.Quantity:
    """Power of a Distance by redispatching to Quantity.

    Examples
    --------
    >>> import math
    >>> from coordinax.distance import Distance

    >>> q1 = Distance(10.0, "m")
    >>> y = 3.0
    >>> q1 ** y
    Quantity['volume'](Array(1000., dtype=float32, ...), unit='m3')

    """
    return u.Quantity(x.value, x.unit) ** y  # TODO: better call to power


# ==============================================================================


@register(lax.sqrt_p)
def _sqrt_p_d(x: AbstractDistance) -> u.Quantity:
    """Square root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from coordinax.distance import Distance
    >>> q = Distance(9, "m")
    >>> jnp.sqrt(q)
    Quantity['m0.5'](Array(3., dtype=float32, ...), unit='m(1/2)')

    >>> from coordinax.distance import Parallax
    >>> q = Parallax(9, "mas")
    >>> jnp.sqrt(q)
    Quantity['rad0.5'](Array(3., dtype=float32, ...), unit='mas(1/2)')

    """
    # Promote to something that supports sqrt units.
    return u.Quantity(lax.sqrt(x.value), unit=x.unit ** (1 / 2))


# ==============================================================================


def _to_value_rad_or_one(q: AbstractQuantity) -> ArrayLike:
    return u.ustrip(radian if u.is_unit_convertible(q.unit, radian) else one, q)


# TODO: figure out a promotion alternative that works in general
@register(lax.tan_p)
def _tan_p_d(x: AbstractDistance) -> u.Quantity["dimensionless"]:
    return u.Quantity(lax.tan(_to_value_rad_or_one(x)), unit=one)
