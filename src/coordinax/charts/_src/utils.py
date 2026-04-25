"""Chart utility functions."""

__all__ = ()

import inspect

from jaxtyping import ArrayLike
from typing import Any, Final, TypeVar, cast

import unxt as u
from unxt.quantity import BareQuantity

V = TypeVar("V", bound=u.AbstractQuantity | ArrayLike)
RAD: Final = u.unit("rad")
ANGLE: Final = u.dimension("angle")


def uconvert_to_rad(value: V, usys: u.AbstractUnitSystem | None, /) -> V:
    """Convert an angle value to radians, handling no-usys case.

    Examples
    --------
    Angular quantities are converted from their own unit:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.charts._src.utils import uconvert_to_rad
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

    """
    from_unit = RAD if usys is None else usys["angle"]
    is_qty = isinstance(value, u.AbstractQuantity)
    source_unit = (
        u.unit_of(value) if is_qty and u.dimension_of(value) == ANGLE else from_unit
    )
    raw = u.uconvert_value(RAD, source_unit, value.value if is_qty else value)  # ty: ignore[possibly-missing-attribute]
    return cast("V", BareQuantity(raw, unit=RAD) if is_qty else raw)


def is_abstract_class(cls: type, /) -> bool:
    """Determine if a class is abstract."""
    return inspect.isabstract(cls) or cls.__name__.startswith("Abstract")


def is_not_abstract_chart_subclass(cls: type[Any], /) -> bool:
    """Check if cls is a non-abstract non-subclass of AbstractChart."""
    from .base import AbstractChart  # noqa: PLC0415

    return not is_abstract_class(cls) and not issubclass(cls, AbstractChart)
