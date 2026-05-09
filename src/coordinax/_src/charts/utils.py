"""Chart utility functions."""

__all__ = ()

import inspect

from jaxtyping import ArrayLike
from typing import Any, Final, overload

import unxt as u
from unxt.quantity import AllowValue, BareQuantity

from .custom_types import OptUSys

RAD: Final = u.unit("rad")
ANGLE: Final = u.dimension("angle")
UNTLS: Final = u.unit("")
DMLS: Final = u.dimension_of(UNTLS)


@overload
def uconvert_to_rad(value: ArrayLike, usys: OptUSys, /) -> ArrayLike: ...
@overload
def uconvert_to_rad(value: u.AbstractQuantity, usys: OptUSys, /) -> BareQuantity: ...
def uconvert_to_rad(value: Any, usys: OptUSys, /) -> Any:
    """Convert an angle value to radians, handling no-usys case.

    Examples
    --------
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
    return BareQuantity(raw_rad, unit=RAD) if unit is not None else raw_rad


def is_abstract_class(cls: type, /) -> bool:
    """Determine if a class is abstract."""
    return inspect.isabstract(cls) or cls.__name__.startswith("Abstract")


def is_not_abstract_chart_subclass(cls: type[Any], /) -> bool:
    """Check if cls is a non-abstract non-subclass of AbstractChart."""
    from .base import AbstractChart  # noqa: PLC0415

    return not is_abstract_class(cls) and not issubclass(cls, AbstractChart)
