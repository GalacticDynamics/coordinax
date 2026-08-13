"""The shared component-key/component-dict vocabulary.

Canonical home for ``CKey`` and ``CDict``. ``coordinaxs.api`` is the root of
the workspace -- every other package depends on it, directly or through
``coordinax``, and it depends on none of them -- so it is the one place all
layers can import from. ``coordinax._src.custom_types`` re-exports these.

``coordinaxs.api`` has no ``__init__.py`` by design, so this module *is* the
public path::

    from coordinaxs.api.custom_types import CDict, CKey

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
