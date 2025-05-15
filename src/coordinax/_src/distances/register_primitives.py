"""Register Quantity support for jax primitives."""
# pylint: disable=import-error, too-many-lines

__all__: list[str] = []

from typing import Any, TypeVar

from jax import lax
from jaxtyping import ArrayLike
from quax import register

import unxt as u
from unxt.quantity import BareQuantity

from .base import AbstractDistance

T = TypeVar("T")

one = u.unit("")
radian = u.unit("radian")


# TODO: can this be done with promotion/conversion instead?
@register(lax.cbrt_p)
def cbrt_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> BareQuantity:
    """Cube root of a distance.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.distance import Distance
    >>> d = Distance(8, "m")
    >>> jnp.cbrt(d)
     BareQuantity(Array(2., dtype=float32, ...), unit='m(1/3)')

    """
    value = lax.cbrt_p.bind(x.value, accuracy=accuracy)
    return BareQuantity(value, unit=x.unit ** (1 / 3))


# ==============================================================================


@register(lax.div_p)
def div_p_abstractdistances(
    x: AbstractDistance, y: AbstractDistance, /
) -> BareQuantity:
    """Division of two Distances.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.distance import Distance

    >>> q1 = Distance(2, "m")
    >>> q2 = Distance(4, "m")
    >>> jnp.divide(q1, q2)
    BareQuantity(Array(0.5, dtype=float32, ...), unit='')

    """
    return BareQuantity(lax.div(x.value, y.value), unit=x.unit / y.unit)


# ==============================================================================


@register(lax.dot_general_p)
def dot_general_p_abstractdistances(
    lhs: AbstractDistance, rhs: AbstractDistance, /, **kwargs: Any
) -> BareQuantity:
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
    BareQuantity(Array(32, dtype=int32), unit='m2')
    >>> q1 @ q2
    BareQuantity(Array(32, dtype=int32), unit='m2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = jnp.asarray([[0, -1,  0], [1,  0,  0], [0,  0,  1]])
    >>> q = u.Quantity([1, 0, 0], "m")
    >>> Rz @ q
    Quantity(Array([0, 1, 0], dtype=int32), unit='m')

    This uses `matmul` for quantities.

    >>> jnp.linalg.matmul(Rz, q)
    Quantity(Array([0, 1, 0], dtype=int32), unit='m')

    """
    value = lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs)
    return BareQuantity(value, unit=lhs.unit * rhs.unit)


# ==============================================================================


@register(lax.integer_pow_p)
def integer_pow_p_abstractdistance(x: AbstractDistance, /, *, y: Any) -> BareQuantity:
    """Integer power of a Distance.

    Examples
    --------
    >>> from coordinax.distance import Distance
    >>> q = Distance(2, "m")
    >>> q ** 3
     BareQuantity(Array(8, dtype=int32, ...), unit='m3')

    """
    return BareQuantity(lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


@register(lax.pow_p)
def pow_p_abstractdistance_arraylike(
    x: AbstractDistance, y: ArrayLike, /
) -> BareQuantity:
    """Power of a Distance by redispatching to Quantity.

    Examples
    --------
    >>> import math
    >>> from coordinax.distance import Distance

    >>> q1 = Distance(10.0, "m")
    >>> y = 3.0
    >>> q1 ** y
    BareQuantity(Array(1000., dtype=float32, ...), unit='m3')

    """
    return BareQuantity(x.value, x.unit) ** y  # TODO: better call to power


# ==============================================================================


@register(lax.sqrt_p)
def sqrt_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> BareQuantity:
    """Square root of a quantity.

    Examples
    --------
    >>> import quaxed.numpy as jnp

    >>> from coordinax.distance import Distance
    >>> q = Distance(9, "m")
    >>> jnp.sqrt(q)
    BareQuantity(Array(3., dtype=float32, ...), unit='m(1/2)')

    >>> from coordinax.distance import Parallax
    >>> q = Parallax(9, "mas")
    >>> jnp.sqrt(q)
    BareQuantity(Array(3., dtype=float32, ...), unit='mas(1/2)')

    """
    # Promote to something that supports sqrt units.
    value = lax.sqrt_p.bind(x.value, accuracy=accuracy)
    return BareQuantity(value, unit=x.unit ** (1 / 2))


# ==============================================================================


def to_value_rad_or_one(q: u.AbstractQuantity, /) -> ArrayLike:
    return u.ustrip(radian if u.is_unit_convertible(q.unit, radian) else one, q)


# TODO: figure out a promotion alternative that works in general
@register(lax.tan_p)
def tan_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> BareQuantity:
    value = lax.tan_p.bind(to_value_rad_or_one(x), accuracy=accuracy)
    return BareQuantity(value, unit=one)
