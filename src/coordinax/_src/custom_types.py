"""Custom types."""

__all__: tuple[str, ...] = (
    "Ang",
    "Len",
    "Spd",
    "OptUSys",
    "CKey",
    "CDict",
    "CDictT",
    "Ks",
    "Ds",
)

from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from typing_extensions import TypeVar

import unxt as u

# =========================================================
# Unit-related Types

# Specific Dimensions
Ang: TypeAlias = Literal["angle"]
Len: TypeAlias = Literal["length"]
Spd: TypeAlias = Literal["speed"]

# Units
OptUSys: TypeAlias = u.AbstractUnitSystem | None

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
CDictT = TypeVar("CDictT", bound=CDict)

Ks = TypeVar("Ks", bound=tuple[CKey, ...], default=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...], default=tuple[str | None, ...])
