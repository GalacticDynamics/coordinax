"""Chart utility functions."""

__all__ = ()

import inspect

from jaxtyping import ArrayLike
from typing import Any, Final, TypeVar, cast

import unxt as u

V = TypeVar("V", bound=u.AbstractQuantity | ArrayLike)
RAD: Final = u.unit("rad")


def uconvert_to_rad(value: V, usys: u.AbstractUnitSystem | None, /) -> V:
    """Convert an angle value to radians, handling no-usys case."""
    out = u.uconvert_value(RAD, RAD if usys is None else usys["angle"], value)
    return cast("V", out)


def is_abstract_class(cls: type, /) -> bool:
    """Determine if a class is abstract."""
    return inspect.isabstract(cls) or cls.__name__.startswith("Abstract")


def is_not_abstract_chart_subclass(cls: type[Any], /) -> bool:
    """Check if cls is a non-abstract non-subclass of AbstractChart."""
    from .base import AbstractChart  # noqa: PLC0415

    return not is_abstract_class(cls) and not issubclass(cls, AbstractChart)
