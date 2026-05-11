"""Custom types for coordinax.ops."""

__all__ = ("Shape", "HasShape", "OptUSys", "CKey", "CDict")

from typing import Any, Protocol, TypeAlias, runtime_checkable

import unxt as u

# =========================================================
# Array-related Types

Shape: TypeAlias = tuple[int, ...]


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover


# =========================================================
# Vector-related Types

OptUSys: TypeAlias = u.AbstractUnitSystem | None

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]
