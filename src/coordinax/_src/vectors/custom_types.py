"""Custom types for vectors."""

__all__ = ("HasShape", "Dimension", "DimensionLike", "Shape", "PDict")


from typing import Any, Protocol, TypeAlias, runtime_checkable

from astropy.units import PhysicalType as Dimension

Shape: TypeAlias = tuple[int, ...]
PDict: TypeAlias = dict[str, Any]

DimensionLike: TypeAlias = Dimension | str


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover
