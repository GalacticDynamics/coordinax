"""Internal custom types for coordinax."""

__all__ = (
    # Dimension-related
    "Ang",
    "Len",
    "Spd",
    # Units-related
    "OptUSys",
    # Array-related
    "Shape",
    "Ks",
    "Ds",
    # Vector-related
    "V",
    "CKey",
    "CDict",
)

from typing import Any, Literal, TypeAlias
from typing_extensions import TypeVar

import unxt as u

#   Specific Dimensions
Ang: TypeAlias = Literal["angle"]
Len: TypeAlias = Literal["length"]
Spd: TypeAlias = Literal["speed"]


# Units
OptUSys: TypeAlias = u.AbstractUnitSystem | None

# =========================================================
# Array-related Types

Shape: TypeAlias = tuple[int, ...]

# =========================================================
# Vector-related Types

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]

# Component Value Type
V = TypeVar("V", default=Any)

Ks = TypeVar("Ks", bound=tuple[CKey, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
