"""Register Quantity support for jax primitives."""
# pylint: disable=import-error

__all__: tuple[str, ...] = ()

from jaxtyping import ArrayLike
from typing import Any

import jax.numpy as jnp
import quax
from jax import lax
from plum import convert
from quax import register

import quaxed.lax as qlax
import unxt as u

from .base import AbstractDistance


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
    return qlax.atan2(convert(x, u.Q), convert(y, u.Q))  # ty: ignore[invalid-return-type]


# ==============================================================================


@register(lax.add_p)
def add_p_abstractdistances(
    x: AbstractDistance, y: AbstractDistance, /
) -> AbstractDistance:
    """Sum of two distance-like quantities, skipping a guard that cannot fire.

    Addition is the one binary operation the non-negative types are closed
    under -- but only when both operands really are non-negative. That is a
    theorem exactly when each *was validated*, so the bypass is gated on it.

    ``check_negative=False`` opts an instance out of validation, and such an
    instance may hold a negative. Adding two of those can produce a negative
    sum, so they fall back to ordinary construction, carrying the left
    operand's setting. Otherwise the result would be an instance whose
    ``check_negative=True`` asserts an invariant its value violates -- and a
    lying `Distance` is worse than a slow one.

    >>> from coordinax.distances import Distance

    >>> Distance(1, "m") + Distance(2, "m")
    Distance(3, 'm')

    The left operand fixes the unit, as before:

    >>> Distance(1, "m") + Distance(2, "km")
    Distance(2001., 'm')

    Opted-out operands keep the opt-out rather than being relabelled:

    >>> a = Distance(-1, "m", check_negative=False)
    >>> a + a
    Distance(-2, 'm', check_negative=False)

    And a validated operand still refuses to absorb an unvalidated negative:

    >>> try: Distance(1, "m") + Distance(-5, "m", check_negative=False)
    ... except Exception as e: print(type(e).__name__)
    EquinoxRuntimeError

    Mismatched dimensions remain a unit error, not a silent result:

    >>> from coordinaxs.astro import Parallax
    >>> try: Distance(1, "m") + Parallax(1, "mas")
    ... except Exception as e: print(type(e).__name__)
    UnitConversionError

    The bypass also rests on non-negativity being the only invariant in this
    hierarchy. A subclass constraining something addition does not preserve
    would need to register its own narrower rule.
    """
    yv = u.ustrip(x.unit, y)
    total = jnp.add(x.value, yv)

    # `None` for a kind with no sign guard at all (e.g. `DistanceModulus`),
    # where there is nothing for the bypass to be unsound about.
    x_validated = getattr(x, "check_negative", None)
    y_validated = getattr(y, "check_negative", None)

    if x_validated is None or (x_validated and y_validated):
        return type(x)._make(total, x.unit)

    # `x_validated is not None` above already established that this kind takes
    # `check_negative`; ty cannot narrow `type(x)` through a `getattr` guard.
    return type(x)(total, x.unit, check_negative=x_validated)  # ty: ignore[unknown-argument]


# ==============================================================================

#: Quaxified bind, matching how `unxt` implements its own `mul_p` rules so the
#: primitive's parameters (e.g. `out_dtype`) are forwarded rather than dropped.
_mul_qbind = quax.quaxify(lax.mul_p.bind)


@register(lax.mul_p)
def mul_p_abstractdistance_arraylike(
    x: AbstractDistance, y: ArrayLike, /, **kw: Any
) -> u.Q:
    """Scaling a non-negative quantity degrades it to a `Quantity`.

    The result is a distance only when the scalar is non-negative, and a
    scalar's sign is not knowable at trace time. That leaves three possible
    behaviours -- raise, return a `Distance` holding a negative value, or widen
    -- and only widening is both total and honest, so this always widens.

    >>> from coordinax.distances import Distance
    >>> Distance(3, "m") * 2
    Q(6, 'm')

    >>> Distance(3, "m") * -1
    Q(-3, 'm')

    """
    return u.Q(_mul_qbind(u.ustrip(x), y, **kw), x.unit)


@register(lax.mul_p)
def mul_p_arraylike_abstractdistance(
    x: ArrayLike, y: AbstractDistance, /, **kw: Any
) -> u.Q:
    """Scaling from the left, as above.

    >>> from coordinax.distances import Distance
    >>> 2 * Distance(3, "m")
    Q(6, 'm')

    """
    return u.Q(_mul_qbind(x, u.ustrip(y), **kw), y.unit)


@register(lax.mul_p)
def mul_p_abstractdistances(
    x: AbstractDistance, y: AbstractDistance, /, **kw: Any
) -> u.Q:
    """Multiply two distances, giving an area rather than a distance.

    >>> from coordinax.distances import Distance
    >>> Distance(2, "m") * Distance(3, "m")
    Q(6, 'm2')

    """
    return u.Q(_mul_qbind(u.ustrip(x), u.ustrip(y), **kw), x.unit * y.unit)


# ==============================================================================


@register(lax.sub_p)
def sub_p_abstractdistances(x: AbstractDistance, y: AbstractDistance, /) -> u.Q:
    """Subtract two non-negative quantities, widening to a `Quantity`.

    Subtraction is not closed on ``[0, inf)``, and which way round a given pair
    falls is a property of the values, not the types. Preserving the type would
    make ``d1 - d2`` succeed or raise depending on the data -- survivable
    eagerly, useless under `jax.jit`, and poisonous under `jax.vmap`, where one
    negative element would fail the whole batch.

    Widening also makes this agree with ``Distance - Quantity``, which already
    returned a `Quantity`; the same operation no longer depends on how the
    right-hand operand happens to be typed.

    >>> from coordinax.distances import Distance

    >>> Distance(3, "m") - Distance(1, "m")
    Q(2, 'm')

    >>> Distance(1, "m") - Distance(3, "m")
    Q(-2, 'm')

    The left operand fixes the unit:

    >>> Distance(1, "km") - Distance(500, "m")
    Q(0.5, 'km')

    """
    xv: Any = u.ustrip(x)
    yv: Any = u.ustrip(x.unit, y)
    return u.Q(xv - yv, x.unit)


# ==============================================================================


@register(lax.div_p)
def div_p_abstractdistance_arraylike(x: AbstractDistance, y: ArrayLike, /) -> u.Q:
    """Dividing by a scalar degrades, for the reason scaling does.

    >>> from coordinax.distances import Distance
    >>> Distance(6, "m") / 2
    Q(3., 'm')

    >>> Distance(6, "m") / -2
    Q(-3., 'm')

    """
    xv: Any = u.ustrip(x)
    return u.Q(xv / y, x.unit)


@register(lax.div_p)
def div_p_arraylike_abstractdistance(x: ArrayLike, y: AbstractDistance, /) -> u.Q:
    """Divide by a distance, giving a reciprocal length.

    >>> from coordinax.distances import Distance
    >>> 1 / Distance(2, "m")
    Q(0.5, '1 / m')

    """
    return u.Q(lax.div(x, u.ustrip(y)), 1 / y.unit)


# ==============================================================================


@register(lax.cbrt_p)
def cbrt_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> u.Q:
    """Cube root of a distance.

    >>> import quaxed.numpy as jnp
    >>> from coordinax.distances import Distance
    >>> d = Distance(8, "m")
    >>> jnp.cbrt(d)
    Q(2., 'm(1/3)')

    """
    return qlax.cbrt(convert(x, u.Q), accuracy=accuracy)  # ty: ignore[invalid-return-type]


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
) -> u.Q:
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
    return u.Q(value, unit=lhs.unit * rhs.unit)


# ==============================================================================


@register(lax.integer_pow_p)
def integer_pow_p_abstractdistance(x: AbstractDistance, /, *, y: Any) -> u.Q:
    """Integer power of a Distance.

    >>> from coordinax.distances import Distance
    >>> q = Distance(2, "m")
    >>> q ** 3
    Q(8, 'm3')

    """
    return qlax.integer_pow(convert(x, u.Q), y)  # ty: ignore[invalid-return-type]


# ==============================================================================


@register(lax.neg_p)
def neg_p_abstractdistance(x: AbstractDistance, /) -> u.Q:
    """Negation of a distance-like quantity degrades to a Quantity.

    `Distance` and `Parallax` are non-negative by construction, so their
    negation is not a value of the same type -- it has to degrade.

    >>> from coordinax.distances import Distance
    >>> q = Distance(10, "m")
    >>> -q
    Q(-10, 'm')

    This holds for every sign-constrained subclass, not just `Distance`:

    >>> from coordinaxs.astro import Parallax
    >>> -Parallax(1, "mas")
    Q(-1, 'mas')

    `DistanceModulus` overrides this: its domain is all of the reals, so
    negation stays closed and it keeps its own type.

    """
    return u.Q(-x.value, x.unit)


# ==============================================================================


@register(lax.pow_p)
def pow_p_abstractdistance_arraylike(x: AbstractDistance, y: ArrayLike, /) -> u.Q:
    """Power of a Distance by redispatching to Quantity.

    >>> import math
    >>> from coordinax.distances import Distance

    >>> q1 = Distance(10.0, "m")
    >>> y = 3.0
    >>> q1 ** y
    Q(1000., 'm3')

    """
    return convert(x, u.Q) ** y


# ==============================================================================


@register(lax.sqrt_p)
def sqrt_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> u.Q:
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
    return qlax.sqrt(convert(x, u.Q), accuracy=accuracy)  # ty: ignore[invalid-return-type]


# ==============================================================================


@register(lax.tan_p)
def tan_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> u.Q:
    """Tangent of a distance kind, degrading to a quantity.

    Only the angle-valued kinds have a tangent; `Distance` and `DistanceModulus`
    raise, as they would as plain quantities.

    >>> import quaxed.numpy as jnp
    >>> from coordinaxs.astro import Parallax

    >>> jnp.tan(Parallax(9, "mas"))
    Q(4.3633...e-08, '')

    """
    return qlax.tan(convert(x, u.Q), accuracy=accuracy)  # ty: ignore[invalid-return-type]
