"""Register Angle support for jax primitives."""
# pylint: disable=import-error

__all__: list[str] = []

from typing import Any, TypeVar

from jax import lax
from jaxtyping import ArrayLike
from plum import convert, promote
from quax import quaxify, register

import unxt as u
from quaxed import lax as qlax
from unxt.quantity import BareQuantity as FastQ

from .base import AbstractAngle

T = TypeVar("T")

one = u.unit("")
radian = u.unit("radian")


# TODO: can this be done with promotion/conversion/default rule instead?
@register(lax.cbrt_p)
def cbrt_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> FastQ:
    """Cube root of an angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(8, "rad")
    >>> jnp.cbrt(q)
    BareQuantity(Array(2., dtype=float32, ...), unit='rad(1/3)')

    """
    return quaxify(lax.cbrt_p.bind)(  # TODO: move to quaxed
        convert(x, FastQ), **kw
    )


# ==============================================================================


# TODO: can this be done with promotion/conversion/default rule instead?
@register(lax.div_p)
def div_p_q_a(x: AbstractAngle, y: AbstractAngle, /) -> u.Quantity:
    """Division of a Quantity by an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.angle import Angle

    >>> angle = Angle(1, "deg")
    >>> q = u.Quantity(2, "km")
    >>> jnp.divide(q, angle)
    Quantity(Array(2., dtype=float32, ...), unit='km / deg')

    """
    x, y = promote(x, y)
    return qlax.div(convert(x, u.Quantity), convert(y, u.Quantity))


# ==============================================================================


@register(lax.cos_p)
def cos_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> FastQ:
    """Cosine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(0, "deg")
    >>> jnp.cos(q)
    BareQuantity(Array(1., dtype=float32, ...), unit='')

    """
    return quaxify(lax.cos_p.bind)(  # TODO: move to quaxed
        convert(x, FastQ), **kw
    )


# ==============================================================================


@register(lax.dot_general_p)
def dot_general_abstractangle_abstractangle(
    lhs: AbstractAngle, rhs: AbstractAngle, /, **kwargs: Any
) -> FastQ:
    """Dot product of two Angles.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q1 = Angle([1, 2, 3], "deg")
    >>> q2 = Angle([4, 5, 6], "deg")
    >>> jnp.vecdot(q1, q2)
    BareQuantity(Array(32, dtype=int32), unit='deg2')

    >>> q1 @ q2
    BareQuantity(Array(32, dtype=int32), unit='deg2')

    """
    value = lax.dot_general_p.bind(lhs.value, rhs.value, **kwargs)
    return FastQ(value, unit=lhs.unit * rhs.unit)


# ==============================================================================


@register(lax.integer_pow_p)
def integer_pow_p_abstractangle(x: AbstractAngle, /, *, y: Any) -> FastQ:
    """Integer power of an Angle.

    Examples
    --------
    >>> from coordinax.angle import Angle
    >>> q = Angle(2, "deg")

    >>> q ** 3
    BareQuantity(Array(8, dtype=int32, ...), unit='deg3')

    """
    return qlax.integer_pow(convert(x, FastQ), y)


# ==============================================================================


@register(lax.pow_p)
def pow_p_abstractangle_arraylike(x: AbstractAngle, y: ArrayLike, /) -> FastQ:
    """Power of an Angle by redispatching to Quantity.

    Examples
    --------
    >>> import math
    >>> from coordinax.angle import Angle

    >>> q1 = Angle(10.0, "deg")
    >>> y = 3.0
    >>> q1 ** y
    BareQuantity(Array(1000., dtype=float32, ...), unit='deg3')

    """
    return qlax.pow(convert(x, FastQ), y)


# ==============================================================================


@register(lax.sin_p)
def sin_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> FastQ:
    """Sine of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(90, "deg")
    >>> jnp.sin(q)
    BareQuantity(Array(1., dtype=float32, ...), unit='')

    """
    return quaxify(lax.sin_p.bind)(  # TODO: move to quaxed
        convert(x, FastQ), **kw
    )


# ==============================================================================


@register(lax.sqrt_p)
def sqrt_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> FastQ:
    """Square root of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(9, "deg")
    >>> jnp.sqrt(q)
    BareQuantity(Array(3., dtype=float32, ...), unit='deg(1/2)')

    """
    return quaxify(lax.sqrt_p.bind)(convert(x, FastQ), **kw)


# ==============================================================================


@register(lax.tan_p)
def tan_p_abstractangle(x: AbstractAngle, /, **kw: Any) -> FastQ:
    """Tangent of an Angle.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from coordinax.angle import Angle

    >>> q = Angle(45, "deg")
    >>> jnp.tan(q)
    BareQuantity(Array(1., dtype=float32, ...), unit='')

    """
    return quaxify(lax.tan_p.bind)(  # TODO: move to quaxed
        convert(x, FastQ), **kw
    )
