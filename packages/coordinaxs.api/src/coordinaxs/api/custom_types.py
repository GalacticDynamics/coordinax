"""The shared component-key/component-dict vocabulary.

Defined here, at the workspace root, so every layer can import it;
``coordinax._src.custom_types`` re-exports.
"""

__all__ = ("CKey", "CDict")

from typing import TYPE_CHECKING, Any, TypeAlias

# Component key type: string for all charts (including dot-delimited product keys)
CKey: TypeAlias = str

if TYPE_CHECKING:
    CDict: TypeAlias = dict[CKey, Any]
else:
    # A parametric `dict[...]` annotation makes every plum signature
    # using CDict "unfaithful", disabling plum's method cache (a full
    # ~200x slower resolution per call). The bare `dict` keeps the cache;
    # the TYPE_CHECKING branch above preserves the static type.
    CDict: TypeAlias = dict
