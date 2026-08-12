"""Custom types.

The one deliberate copy of `coordinax`'s type vocabulary: ``coordinaxs.api``
must not depend on ``coordinax``, so it cannot re-export from
``coordinax.internal`` the way every other package does. Keep the `CDict`
definition below in step with ``coordinax._src.custom_types``.
"""

__all__ = ("CKey", "CDict")

from typing import TYPE_CHECKING, Any, TypeAlias

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
