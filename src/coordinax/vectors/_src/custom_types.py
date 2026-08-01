"""Custom types for coordinax.vectors."""

__all__ = ("Shape", "HasShape", "CKey", "CDict", "OptUSys")

from typing import Protocol, TypeAlias, runtime_checkable

from coordinax._src.custom_types import CDict, CKey, OptUSys

Shape: TypeAlias = tuple[int, ...]


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover
