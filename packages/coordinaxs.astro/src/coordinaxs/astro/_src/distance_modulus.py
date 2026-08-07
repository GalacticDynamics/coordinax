"""Distance quantities."""

__all__ = ("DistanceModulus",)


from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, final

import equinox as eqx
import plum
import quax
from jax import lax
from quax import register

import quaxed.numpy as jnp
import unxt as u

import coordinax.distances as cxd
from .constants import ANGLE, LENGTH, MAGNITUDE

#: Matches how `unxt` binds `mul_p`, forwarding the primitive's parameters.
_mul_qbind = quax.quaxify(lax.mul_p.bind)

parallax_base_length = u.Q(jnp.array(1), "AU")

#: The distance-modulus zero point: dm == 0 at d == 10 pc, by definition.
#: An `int`, so an integer-dtype distance keeps the dtype the pre-#634 form gave
#: it: under ``JAX_ENABLE_X64`` an int32 divided by a weak float widens to
#: float64, by a weak int stays float32. Float inputs are unaffected either way.
_DM_ZERO_POINT_PC = 10


def _distance_modulus_from_pc(d_pc: ArrayLike, /) -> ArrayLike:
    """Distance modulus in mag, from a distance already stripped to parsecs.

    ``dm = 5 log10(d / 10 pc)`` -- the textbook form, scaled inside the log
    rather than corrected after it.

    The algebraically equal ``5 * log10(d_pc) - 5`` is less accurate near the
    zero point. ``log10(d_pc)`` carries a relative error, so for d ~ 10 pc
    (where log10 ~ 1) ``5 * log10(...)`` carries an absolute error of ~5 eps;
    the following ``- 5`` is exact by Sterbenz and so cannot recover it, while
    the result itself is heading to zero. Scaling the argument instead leaves
    only the ~eps/ln(10) of the single division, ~2 eps after the factor of 5.

    Worst-case absolute error over d in {9, 9.9, ..., 50} pc, float32,
    measured through this code path (``ustrip`` contributes, so an isolated
    snippet reads several orders low):

    ==============================  =========  =========
    form                            abs. err   runtime
    ==============================  =========  =========
    ``5 * log10(d_pc) - 5``          4.48e-07     1.000x
    this form                        1.37e-07     0.985x
    ``5 * (log10(d_pc) - 1)``        4.66e-07     1.000x
    ``(5/ln10) * log1p((d-10)/10)``  1.01e-07     1.980x
    ==============================  =========  =========

    3.3x more accurate at no cost. XLA strength-reduces the division to a
    multiply, so this lowers to the same three f32 ops as the original
    (multiply, log, multiply); over 60 paired in-process rounds it came out
    marginally ahead and won 31 of them, i.e. indistinguishable.

    Not uniformly better: at d = 9 pc this form carries 7.4e-08 against the
    subtractive form's 3.1e-08. The gain is concentrated where dm -> 0, which
    is where absolute error matters and where the subtractive form is worst.

    The ``log1p`` form is better still -- and much better in the 9.9-10.1 pc
    band -- but is genuinely 2x slower, so it is not used here.
    """
    return 5 * jnp.log10(d_pc / _DM_ZERO_POINT_PC)


@final
class DistanceModulus(cxd.AbstractDistance):
    """Distance modulus quantity.

    Examples
    --------
    >>> from coordinaxs.astro import DistanceModulus
    >>> DistanceModulus(10, "mag")
    DistanceModulus(10, 'mag')

    The units are checked to have magnitude dimensions.

    >>> try: DistanceModulus(10, "pc")
    ... except ValueError as e: print(e)
    Distance modulus must have units of magnitude.

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)  # ty: ignore[invalid-assignment]
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit != u.unit("mag"):
            msg = "Distance modulus must have units of magnitude."
            raise ValueError(msg)


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[DistanceModulus], value: ArrayLike, unit: Any, /, **kw: Any
) -> DistanceModulus:
    """Construct a distance.

    >>> import unxt as u
    >>> from coordinaxs.astro import DistanceModulus

    >>> DistanceModulus.from_(1, "mag")
    DistanceModulus(1, 'mag')

    """
    return cls(jnp.asarray(value, **kw), unit)


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[DistanceModulus], dm: DistanceModulus, /, **kw: Any
) -> DistanceModulus:
    """Compute distance modulus from distance modulus.

    >>> import unxt as u
    >>> from coordinaxs.astro import DistanceModulus

    >>> dm = DistanceModulus(1, "mag")
    >>> DistanceModulus.from_(dm) is dm
    True

    >>> DistanceModulus.from_(dm, dtype=float)
    DistanceModulus(1., 'mag')

    """
    if len(kw) == 0:
        return dm
    return jnp.asarray(dm, **kw)  # ty: ignore[invalid-return-type]


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[DistanceModulus], q: u.AbstractQuantity, /, **kw: Any
) -> DistanceModulus:
    """Construct a distance modulus from a quantity, dispatching on dimensions.

    From a distance:

    >>> import unxt as u
    >>> import coordinax.distances as cxd
    >>> from coordinaxs.astro import DistanceModulus

    >>> d = cxd.Distance(1, "pc")
    >>> DistanceModulus.from_(d)
    DistanceModulus(-5., 'mag')

    >>> q = u.Q(1, "pc")
    >>> DistanceModulus.from_(q)
    DistanceModulus(-5., 'mag')

    From a parallax angle:

    >>> q = u.Q(1, "mas")
    >>> DistanceModulus.from_(q)
    DistanceModulus(10., 'mag')

    From a distance modulus (magnitude):

    >>> q = u.Q(1, "mag")
    >>> DistanceModulus.from_(q)
    DistanceModulus(1, 'mag')

    """
    dim = u.dimension_of(q)

    if dim == LENGTH:  # distance
        dm = _distance_modulus_from_pc(q.ustrip("pc"))
        return cls(jnp.asarray(dm, **kw), "mag")

    if dim == ANGLE:  # parallax
        d = parallax_base_length / jnp.tan(q)  # [AU]
        dm = _distance_modulus_from_pc(d.ustrip("pc"))
        return cls(jnp.asarray(dm, **kw), "mag")

    if dim == MAGNITUDE:  # already a distance modulus (magnitude)
        unit = u.unit_of(q)
        return cls(jnp.asarray(u.ustrip(unit, q), **kw), unit)

    msg = f"cannot build a DistanceModulus from a quantity with dimension {dim}"
    raise ValueError(msg)


@cxd.Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxd.Distance], dm: DistanceModulus, /, **kw: Any) -> cxd.Distance:
    """Compute distance from distance modulus.

    >>> import coordinax.distances as cxd
    >>> from coordinaxs.astro import DistanceModulus

    >>> dm = DistanceModulus(10, "mag")
    >>> cxd.Distance.from_(dm).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = 10 ** (1 + dm.ustrip("mag") / 5)
    # The guard is free here: 10**x lowers to exp(x*ln10), and XLA folds
    # `exp(...) < 0` to false, eliminating the check entirely.
    return cls(jnp.asarray(d, **kw), "pc")


@register(lax.neg_p)
def neg_p_distancemodulus(x: DistanceModulus, /) -> DistanceModulus:
    """Negation of a distance modulus stays a distance modulus.

    This overrides the `AbstractDistance` rule, which degrades to a `Quantity`
    because `Distance` and `Parallax` are non-negative by construction. A
    distance modulus is not: ``dm = 5 log10(d / 10 pc)`` maps d in (0, inf)
    onto all of the reals, and a negative value is the ordinary way to say
    "nearer than 10 pc". Negation is closed here, so the type survives it.

    >>> from coordinaxs.astro import DistanceModulus
    >>> -DistanceModulus(10, "mag")
    DistanceModulus(-10, 'mag')

    >>> -DistanceModulus(-5, "mag")
    DistanceModulus(5, 'mag')

    """
    return DistanceModulus(-x.value, x.unit)


@register(lax.sub_p)
def sub_p_distancemoduli(x: DistanceModulus, y: DistanceModulus, /) -> DistanceModulus:
    """Subtract two distance moduli, keeping the type.

    Overrides the `AbstractDistance` rule, which widens to a `Quantity` because
    `Distance` and `Parallax` cannot represent a negative. A distance modulus
    can: its domain is all of the reals, so subtraction is closed here.

    >>> from coordinaxs.astro import DistanceModulus
    >>> DistanceModulus(1, "mag") - DistanceModulus(3, "mag")
    DistanceModulus(-2, 'mag')

    """
    yv: Any = u.ustrip(x.unit, y)
    return DistanceModulus(x.value - yv, x.unit)


@register(lax.mul_p)
def mul_p_distancemodulus_arraylike(
    x: DistanceModulus, y: ArrayLike, /, **kw: Any
) -> DistanceModulus:
    """Scaling a distance modulus keeps it a distance modulus, either sign.

    >>> from coordinaxs.astro import DistanceModulus
    >>> DistanceModulus(2, "mag") * 3
    DistanceModulus(6, 'mag')

    >>> DistanceModulus(2, "mag") * -1
    DistanceModulus(-2, 'mag')

    """
    return DistanceModulus(_mul_qbind(u.ustrip(x), y, **kw), x.unit)


@register(lax.mul_p)
def mul_p_arraylike_distancemodulus(
    x: ArrayLike, y: DistanceModulus, /, **kw: Any
) -> DistanceModulus:
    """Scaling from the left, as above.

    >>> from coordinaxs.astro import DistanceModulus
    >>> 3 * DistanceModulus(2, "mag")
    DistanceModulus(6, 'mag')

    """
    return DistanceModulus(_mul_qbind(x, u.ustrip(y), **kw), y.unit)


@register(lax.div_p)
def div_p_distancemodulus_arraylike(
    x: DistanceModulus, y: ArrayLike, /
) -> DistanceModulus:
    """Dividing a distance modulus by a scalar keeps the type.

    >>> from coordinaxs.astro import DistanceModulus
    >>> DistanceModulus(4, "mag") / 2
    DistanceModulus(2., 'mag')

    """
    xv: Any = u.ustrip(x)
    return DistanceModulus(xv / y, x.unit)


@plum.dispatch
def dimension_of(obj: type[DistanceModulus], /) -> u.AbstractDimension:
    """Return the magnitude dimension: a distance modulus is not a length.

    See `coordinaxs.astro._src.parallax.dimension_of` -- same correction, for
    the same reason: the inherited `type[AbstractDistance]` rule reports length,
    while every `DistanceModulus` instance reports magnitude.

    >>> import unxt as u
    >>> from coordinaxs.astro import DistanceModulus
    >>> u.dimension_of(DistanceModulus)
    PhysicalType('unknown')

    Which is what the instances say (astropy has no dedicated physical type for
    magnitude, so it resolves to ``'unknown'``):

    >>> u.dimension_of(DistanceModulus(10, "mag"))
    PhysicalType('unknown')

    """
    return MAGNITUDE
