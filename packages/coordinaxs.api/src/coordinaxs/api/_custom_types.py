__all__ = ("CKey", "CDict")

from typing import Any, TYPE_CHECKING, TypeAlias

# Component key type: string for all charts (including dot-delimited product keys)
CKey: TypeAlias = str

# Parameter dictionary type alias
if TYPE_CHECKING:
    # Typed for static checkers only.
    CDict: TypeAlias = dict[CKey, Any]
else:
    # A parametric `dict[...]` annotation makes every plum signature
    # using CDict "unfaithful", disabling plum's method cache (a full
    # ~200x slower resolution per call). The bare `dict` keeps the cache;
    # the TYPE_CHECKING branch above preserves the static type.
    CDict: TypeAlias = dict
