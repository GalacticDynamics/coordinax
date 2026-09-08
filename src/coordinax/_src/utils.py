"""Chart utility functions."""

__all__ = ()


from jaxtyping import ArrayLike
from typing import Any, Final, overload

import unxt as u
from unxt.quantity import AllowValue, Quantity

from .custom_types import OptUSys
from coordinaxs.api.custom_types import CDict, CKeys

RAD: Final = u.unit("rad")
ANGLE: Final = u.dimension("angle")
UNTLS: Final = u.unit("")
DMLS: Final = u.dimension_of(UNTLS)


@overload
def uconvert_to_rad(value: ArrayLike, usys: OptUSys, /) -> ArrayLike: ...
@overload
def uconvert_to_rad(value: u.AbstractQuantity, usys: OptUSys, /) -> Quantity: ...
def uconvert_to_rad(value: Any, usys: OptUSys, /) -> Any:
    """Convert an angle value to radians, handling no-usys case.

    Angular quantities are converted from their own unit:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> out = uconvert_to_rad(u.Q(90, "deg"), None)
    >>> out.unit == u.unit("rad") and bool(jnp.allclose(out.value, jnp.pi / 2))
    True

    Dimensionless quantities are interpreted in the unit system's angle unit:

    >>> usys = u.unitsystem("m", "deg")
    >>> out = uconvert_to_rad(u.Q(90, ""), usys)
    >>> out.unit == u.unit("rad") and bool(jnp.allclose(out.value, jnp.pi / 2))
    True

    Plain numeric values are treated as radians by default:

    >>> bool(jnp.allclose(uconvert_to_rad(jnp.pi / 2, None), jnp.pi / 2))
    True

    Plain numeric values can also be interpreted through ``usys["angle"]``:

    >>> bool(jnp.allclose(uconvert_to_rad(90.0, usys), jnp.pi / 2))
    True

    Non-angle, non-dimensionless quantities are rejected:

    >>> uconvert_to_rad(u.Q(1, "m"), None)
    Traceback (most recent call last):
        ...
    ValueError: Unsupported quantity dimension for angle conversion: length

    """
    from_unit = RAD if usys is None else usys["angle"]
    unit = u.unit_of(value)
    source_unit = from_unit

    if unit is not None:
        dim = u.dimension_of(unit)
        if dim == ANGLE:
            source_unit = unit
        elif dim != DMLS:
            msg = f"Unsupported quantity dimension for angle conversion: {dim}"
            raise ValueError(msg)

    raw_rad = u.uconvert_value(RAD, source_unit, u.ustrip(AllowValue, value))
    return Quantity(raw_rad, unit=RAD) if unit is not None else raw_rad


# ===================================================================
# Unit bookkeeping for chart transition maps
#
# A `pt_map` body written against `Quantity` operands pays `quax` once per
# arithmetic primitive: `quax` builds and evaluates a jaxpr for each one, so
# eager cost is the *count* of ops rather than the work in them. Measured on a
# scalar point, one primitive is ~4us on a raw array against 120-820us on a
# `Quantity` -- `Quantity ** 2` 120us, `+` 198us, `/` 229us, `== 0` 444us,
# `atan2` 617us, `acos` 623us. `Cart3D -> Spherical3D` spent ~2900us of its
# ~3500us there, which is simply the sum of its ops.
#
# `strip`/`wrap` move the units out of the arithmetic instead: resolve them
# once on the way in, run the body on raw arrays, re-attach once on the way
# out. Same numbers, and the unit-promotion rule is unchanged because
# expressing the group in its *first* component's unit is exactly what
# `Quantity` arithmetic already did -- `{km, m, cm}` yielded `r` in km before
# and still does.
#
# These are deliberately written against `unxt`'s public functions rather than
# reaching for `.unit`/`.value`, which would be ~8x faster again today. That
# gap is `plum` declining to cache the unfaithful signatures on `ustrip` and
# friends, which plum#290 addresses; when it lands the public route gets the
# rest of the win for free, whereas a reach-through would have to be unpicked.
# Measured on a scalar `cart3d -> sph3d`: 3490us before, ~260us here.


def strip(p: CDict, keys: CKeys, /) -> tuple[tuple[Any, ...], Any]:
    """Return the raw values for *keys*, plus the unit they are expressed in.

    All components are converted to the unit of the *first* one, and the unit
    is returned alongside so a caller can re-attach it with `wrap`. When the
    first component carries no unit the values are passed through untouched
    and the unit is `None`, which is how a bare-array point stays a bare-array
    point.

    *keys* must name a **dimensionally homogeneous** group -- the components
    that can meaningfully share a unit. A chart mixing lengths with velocities
    (`PoincarePolar6D`, a phase-space product) needs one call per group, not
    one call per chart. A violation is not silent: `unxt.ustrip` raises
    `UnitConversionError`.

    >>> import unxt as u
    >>> from coordinax._src.utils import strip

    >>> strip({"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}, ("x", "y"))
    ((Array(1., dtype=float64, ...), Array(2., dtype=float64, ...)), Unit("m"))

    Mixed units resolve to the first component's:

    >>> strip({"x": u.Q(1.0, "km"), "y": u.Q(500.0, "m")}, ("x", "y"))
    ((Array(1., dtype=float64, ...), Array(0.5, dtype=float64, ...)), Unit("km"))

    Unitless values pass straight through:

    >>> strip({"x": 1.0, "y": 2.0}, ("x", "y"))
    ((1.0, 2.0), None)

    Incompatible dimensions raise rather than silently reinterpret. `ustrip`
    does this for us, so a group whose components cannot share a unit fails
    loudly at the point of the mistake:

    >>> strip({"x": u.Q(1.0, "m"), "t": u.Q(2.0, "s")}, ("x", "t"))
    Traceback (most recent call last):
        ...
    astropy.units.errors.UnitConversionError: 's' (time) and 'm' (length) are
    not convertible

    """
    unit = u.unit_of(p[keys[0]])
    if unit is None:
        return tuple(p[k] for k in keys), None
    return tuple(u.ustrip(unit, p[k]) for k in keys), unit


def wrap(value: Any, unit: Any, /) -> Any:
    """Re-attach *unit* to a raw value, or pass it through when *unit* is `None`.

    The complement of `strip`: `None` means the point came in unitless and
    goes back out unitless.

    >>> import unxt as u
    >>> from coordinax._src.utils import wrap

    >>> wrap(2.0, u.unit("m"))
    Q(2., 'm')

    >>> wrap(2.0, None)
    2.0

    """
    return value if unit is None else Quantity(value, unit=unit)


def wrap_angle(value: Any, unit: Any, /) -> Any:
    """Wrap a radian-valued raw array, or pass it through when *unit* is `None`.

    *value* is already in radians -- every inverse-trigonometric function
    returns radians -- so *unit* is read only as the "was there a unit at all"
    flag that `strip` returned for the group this angle was derived from.

    A plain `Quantity` is returned rather than an `Angle`;
    `canonical_containers` promotes it, which is where that decision already
    lives and is ~30x cheaper than the checked `Angle` constructor.

    >>> import unxt as u
    >>> from coordinax._src.utils import wrap_angle

    >>> wrap_angle(1.5, u.unit("m"))
    Q(1.5, 'rad')

    >>> wrap_angle(1.5, None)
    1.5

    """
    return value if unit is None else Quantity(value, unit=RAD)
