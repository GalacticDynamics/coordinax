"""Custom types for coordinax.vectors."""

__all__ = ("Shape", "HasShape", "CKey", "CDict")

from typing import Protocol, TypeAlias, runtime_checkable

import unxt as u

CKey: TypeAlias = str
# NOTE: deliberately the bare `dict`, not `dict[str, Any]`: a parametric
# annotation makes every plum signature that uses CDict "unfaithful",
# which disables plum's method cache and forces a full (~200x slower)
# resolution on every call of `act`/`pt_map`/`cconvert`/etc.
CDict: TypeAlias = dict

Shape: TypeAlias = tuple[int, ...]


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover


OptUSys: TypeAlias = u.AbstractUnitSystem | None
