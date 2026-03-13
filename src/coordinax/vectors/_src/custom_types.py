"""Custom types for coordinax.vectors."""

__all__ = ("Shape", "HasShape", "CKey", "CDict")

from typing import Any, Protocol, TypeAlias, runtime_checkable

import unxt as u

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]

Shape: TypeAlias = tuple[int, ...]


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover


OptUSys: TypeAlias = u.AbstractUnitSystem | None
