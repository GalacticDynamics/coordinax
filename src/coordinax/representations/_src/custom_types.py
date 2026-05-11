"""Internal custom types for coordinax."""

__all__ = (
    # Units-related
    "OptUSys",
    # Vector-related
    "CKey",
    "CDict",
)

from typing import Any, TypeAlias

import unxt as u

# =========================================================
# Units-related Types

# Units
OptUSys: TypeAlias = u.AbstractUnitSystem | None

# =========================================================
# Vector-related Types

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]
