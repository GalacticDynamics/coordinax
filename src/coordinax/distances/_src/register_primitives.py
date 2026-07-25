"""Register Quantity support for jax primitives."""
# pylint: disable=import-error, too-many-lines

__all__: tuple[str, ...] = ()

from jaxtyping import ArrayLike
from typing import Any, TypeVar

import plum
from jax import lax
from quax import register

import unxt as u
from unxt.quantity import Quantity

from .base import AbstractDistance
from .constants import ONE, RADIAN
from .measures import Distance

T = TypeVar("T")


# TODO: can this be done with promotion/conversion instead?
@register(lax.atan2_p)
def atan2_p_abstractdistances(x: AbstractDistance, y: AbstractDistance, /) -> u.Q:
    """Arctangent2 of two distances degrades to a quantity.

    >>> import quaxed.numpy as jnp
    >>> from coordinax.distances import Distance

    >>> q1 = Distance(1, "m")
    >>> q2 = Distance(3, "m")
    >>> jnp.atan2(q1, q2)
    Q(0.32175055, 'rad')

    """
    x, y = plum.promote(x, y)  # ty: ignore[too-many-positional-arguments]
    yv = u.ustrip(x.unit, y)
    return u.Q(lax.atan2(u.ustrip(x), yv), unit=RADIAN)


# ==============================================================================


# TODO: can this be done with promotion/conversion instead?
@register(lax.cbrt_p)
def cbrt_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> Quantity:
    """Cube root of a distance.

    >>> import quaxed.numpy as jnp
    >>> from coordinax.distances import Distance
    >>> d = Distance(8, "m")
    >>> jnp.cbrt(d)
    Q(2., 'm(1/3)')

    """
    value = lax.cbrt_p.bind(x.value, accuracy=accuracy)
    return Quantity(value, unit=x.unit ** (1 / 3))


# ==============================================================================


@register(lax.div_p)
def div_p_abstractdistances(x: AbstractDistance, y: AbstractDistance, /) -> u.Q:
    """Division of two Distances.

    >>> import quaxed.numpy as jnp
    >>> from coordinax.distances import Distance

    >>> q1 = Distance(2, "m")
    >>> q2 = Distance(4, "m")
    >>> jnp.divide(q1, q2)
    Q(0.5, '')

    """
    return u.Q(lax.div(x.value, y.value), unit=x.unit / y.unit)


# ==============================================================================


@register(lax.dot_general_p)
def dot_general_p_abstractdistances(
    lhs: AbstractDistance, rhs: AbstractDistance, /, **kwargs: Any
) -> Quantity:
    """Dot product of two Distances.

    This is a dot product of two Distances.

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.distances import Distance

    >>> q1 = Distance([1, 2, 3], "m")
    >>> q2 = Distance([4, 5, 6], "m")
    >>> jnp.vecdot(q1, q2)
    Q(32, 'm2')
    >>> q1 @ q2
    Q(32, 'm2')

    This rule is also used by `jnp.matmul` for quantities.

    >>> Rz = jnp.asarray([[0, -1,  0], [1,  0,  0], [0,  0,  1]])
    >>> q = u.Q([1, 0, 0], "m")
    >>> Rz @ q
    Q([0, 1, 0], 'm')

    This uses `matmul` for quantities.

    >>> jnp.linalg.matmul(Rz, q)
    Q([0, 1, 0], 'm')

    """
    value = lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs)
    return Quantity(value, unit=lhs.unit * rhs.unit)


# ==============================================================================


@register(lax.integer_pow_p)
def integer_pow_p_abstractdistance(x: AbstractDistance, /, *, y: Any) -> Quantity:
    """Integer power of a Distance.

    >>> from coordinax.distances import Distance
    >>> q = Distance(2, "m")
    >>> q ** 3
    Q(8, 'm3')

    """
    return Quantity(lax.integer_pow(x.value, y), unit=x.unit**y)


# ==============================================================================


@register(lax.neg_p)
def neg_p_distance(x: Distance, /) -> u.Q:
    """Negation of a Distance degrades to a Quantity.

    >>> from coordinax.distances import Distance
    >>> q = Distance(10, "m")
    >>> -q
    Q(-10, 'm')

    """
    return u.Q(-x.value, x.unit)


# ==============================================================================


@register(lax.pow_p)
def pow_p_abstractdistance_arraylike(x: AbstractDistance, y: ArrayLike, /) -> Quantity:
    """Power of a Distance by redispatching to Quantity.

    >>> import math
    >>> from coordinax.distances import Distance

    >>> q1 = Distance(10.0, "m")
    >>> y = 3.0
    >>> q1 ** y
    Q(1000., 'm3')

    """
    # TODO: better call to power
    return Quantity(x.value, x.unit) ** y


# ==============================================================================


@register(lax.sqrt_p)
def sqrt_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> Quantity:
    """Square root of a quantity.

    >>> import quaxed.numpy as jnp

    >>> from coordinax.distances import Distance
    >>> q = Distance(9, "m")
    >>> jnp.sqrt(q)
    Q(3., 'm(1/2)')

    >>> from coordinaxs.astro import Parallax
    >>> q = Parallax(9, "mas")
    >>> jnp.sqrt(q)
    Q(3., 'mas(1/2)')

    """
    # Promote to something that supports sqrt units.
    value = lax.sqrt_p.bind(x.value, accuracy=accuracy)
    return Quantity(value, unit=x.unit ** (1 / 2))


# ==============================================================================


def to_value_rad_or_one(q: u.AbstractQuantity, /) -> ArrayLike:
    return u.ustrip(RADIAN if u.is_unit_convertible(q.unit, RADIAN) else ONE, q)  # ty: ignore[invalid-return-type]


# TODO: figure out a promotion alternative that works in general
@register(lax.tan_p)
def tan_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> Quantity:
    value = lax.tan_p.bind(to_value_rad_or_one(x), accuracy=accuracy)
    return Quantity(value, unit=ONE)
