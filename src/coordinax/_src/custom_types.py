"""Custom types.

The single definition of ``coordinax``'s shared type vocabulary; every layer
re-exports from here.

``CKey`` and ``CDict`` come from `coordinaxs.api.custom_types`, the workspace
root, and are re-exported below.
"""

__all__: tuple[str, ...] = (
    "Ang",
    "Len",
    "Spd",
    "OptUSys",
    "Shape",
    "HasShape",
    "CKey",
    "CDict",
    "CDictT",
    "Ks",
    "Ds",
)

from typing import Literal, Protocol, TypeAlias, runtime_checkable
from typing_extensions import TypeVar

import unxt as u

from coordinaxs.api.custom_types import CDict, CKey

# =========================================================
# Unit-related Types

# Specific Dimensions
Ang: TypeAlias = Literal["angle"]
Len: TypeAlias = Literal["length"]
Spd: TypeAlias = Literal["speed"]

# Units
OptUSys: TypeAlias = u.AbstractUnitSystem | None

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

CDictT = TypeVar("CDictT", bound=CDict)

Ks = TypeVar("Ks", bound=tuple[CKey, ...], default=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...], default=tuple[str | None, ...])
