"""Shared utilities for the quantity_matrix package."""

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import unxt as u

if TYPE_CHECKING:
    # Typed for static checkers only.
    CDict: TypeAlias = dict[str, Any]
else:
    # A parametric `dict[...]` annotation makes every plum signature
    # using CDict "unfaithful", disabling plum's method cache (a full
    # ~200x slower resolution per call). The bare `dict` keeps the cache;
    # the TYPE_CHECKING branch above preserves the static type.
    CDict: TypeAlias = dict
_DMLS = u.unit("")

PackedUnitOutput: TypeAlias = tuple[u.AbstractUnit | None, ...]


def strict_zip(*args: Any) -> zip:
    """Zip iterables while enforcing equal lengths."""
    return zip(*args, strict=True)


def cdict_units(p: CDict, keys: tuple[str, ...], /) -> PackedUnitOutput:
    """Extract per-key units from a component dictionary.

    Non-quantity entries yield `None`, so the output tuple can be used for
    heterogeneous dictionaries containing both quantity and non-quantity data.

    >>> import unxt as u
    >>> d = {'x': u.Q(1.0, 'm'), 'y': 2.0, 'z': u.Q(3.0, 'kg')}
    >>> cdict_units(d, ('x', 'y', 'z'))
    (Unit("m"), None, Unit("kg"))

    """
    # `unit_of()` returns None for non-quantities, so this works for both cases.
    return cast("PackedUnitOutput", tuple(u.unit_of(p[k]) for k in keys))
