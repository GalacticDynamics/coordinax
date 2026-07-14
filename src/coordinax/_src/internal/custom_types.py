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
# NOTE: deliberately the bare `dict`, not `dict[str, Any]`: a parametric
# annotation makes every plum signature that uses CDict "unfaithful",
# which disables plum's method cache and forces a full (~200x slower)
# resolution on every call of `act`/`pt_map`/`cconvert`/etc.
CDict: TypeAlias = dict

# Component Value Type
V = TypeVar("V", default=Any)

Ks = TypeVar("Ks", bound=tuple[CKey, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
