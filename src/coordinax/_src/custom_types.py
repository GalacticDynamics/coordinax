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

from typing import Literal, TypeAlias
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
# NOTE: deliberately the bare `dict`, not `dict[str, Any]`: a parametric
# annotation makes every plum signature that uses CDict "unfaithful",
# which disables plum's method cache and forces a full (~200x slower)
# resolution on every call of `act`/`pt_map`/`cconvert`/etc.
CDict: TypeAlias = dict
CDictT = TypeVar("CDictT", bound=CDict)

Ks = TypeVar("Ks", bound=tuple[CKey, ...], default=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...], default=tuple[str | None, ...])
