"""Custom types for coordinax.ops."""

__all__ = ("HasShape", "OptUSys", "CKey", "CDict")

from typing import Any, Protocol, TypeAlias, runtime_checkable

import unxt as u

from coordinax.internal.custom_types import Shape

OptUSys: TypeAlias = u.AbstractUnitSystem | None

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover
