"""Internal custom types for coordinax."""

__all__ = (
    # Units-related
    "OptUSys",
    # Vector-related
    "CKey",
    "CDict",
)

from typing import TYPE_CHECKING, Any, TypeAlias

import unxt as u

# =========================================================
# Units-related Types

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
