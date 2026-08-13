"""The shared component-key/component-dict vocabulary.

Defined here, at the workspace root, so every layer can import it;
``coordinax._src.custom_types`` re-exports.
"""

__all__ = ("CKey", "CKeys", "CDict")

from typing import TYPE_CHECKING, Any, TypeAlias

# Component key type: string for all charts (including dot-delimited product keys)
CKey: TypeAlias = str

if TYPE_CHECKING:
    CDict: TypeAlias = dict[CKey, Any]
    CKeys: TypeAlias = tuple[CKey, ...]
else:
    # A parametric `dict[...]`/`tuple[...]` annotation makes every plum
    # signature using it "unfaithful", disabling plum's method cache (a full
    # ~200x slower resolution per call). The bare containers keep the cache;
    # the TYPE_CHECKING branch above preserves the static types.
    CDict: TypeAlias = dict
    CKeys: TypeAlias = tuple
