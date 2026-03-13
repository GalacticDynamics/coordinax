"""Chart utility functions."""

__all__ = ()

from jaxtyping import ArrayLike
from typing import Any, Final, TypeVar

import unxt as u

from .base import AbstractChart

V = TypeVar("V", bound=u.AbstractQuantity | ArrayLike)
RAD: Final = u.unit("rad")


def uconvert_to_rad(value: V, usys: u.AbstractUnitSystem | None, /) -> V:
    """Convert an angle value to radians, handling no-usys case."""
    return u.uconvert_value(RAD, RAD if usys is None else usys["angle"], value)


def is_not_abstract_chart_subclass(cls: type[Any], /) -> bool:
    """Check if cls is a non-abstract subclass of AbstractChart."""
    return not cls.__name__.startswith("Abstract") and not issubclass(
        cls, AbstractChart
    )
