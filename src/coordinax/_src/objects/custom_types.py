"""Custom types for vectors."""

__all__ = ("HasShape",)


from typing import Protocol, runtime_checkable

from coordinax._src.custom_types import Shape


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover
