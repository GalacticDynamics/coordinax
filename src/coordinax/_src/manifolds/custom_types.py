"""Internal custom types for coordinax."""

__all__ = (
    # Units-related
    "OptUSys",
    # Vector-related
    "CKey",
    "CDict",
    "Ks",
    "Ds",
)

from typing import Any, TypeAlias, TypeVar

import unxt as u

# =========================================================
# Units-related Types

# Units
OptUSys: TypeAlias = u.AbstractUnitSystem | None

# =========================================================
# Vector-related Types

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]

Ks = TypeVar("Ks", bound=tuple[CKey, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
