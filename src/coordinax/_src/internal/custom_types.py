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

from typing import TYPE_CHECKING, Any, Literal, TypeAlias
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
if TYPE_CHECKING:
    # Typed for static checkers only.
    CDict: TypeAlias = dict[CKey, Any]
else:
    # A parametric `dict[...]` annotation makes every plum signature
    # using CDict "unfaithful", disabling plum's method cache (a full
    # ~200x slower resolution per call). The bare `dict` keeps the cache;
    # the TYPE_CHECKING branch above preserves the static type.
    CDict: TypeAlias = dict

# Component Value Type
V = TypeVar("V", default=Any)

Ks = TypeVar("Ks", bound=tuple[CKey, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
