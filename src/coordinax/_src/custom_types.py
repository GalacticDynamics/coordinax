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

from typing import Any, Literal, TypeAlias, TypeVar

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
CDict: TypeAlias = dict[CKey, Any]
CDictT = TypeVar("CDictT", bound=CDict)

Ks = TypeVar("Ks", bound=tuple[CKey, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
