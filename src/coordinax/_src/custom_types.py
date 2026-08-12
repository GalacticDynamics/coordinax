"""Custom types.

This is the single definition of the type vocabulary shared across
``coordinax``. Every layer (``vectors``, ``representations``, ``transforms``,
``_src.internal``) re-exports from here rather than restating the aliases:
`CDict` in particular carries a runtime invariant (see below) that is easy to
break silently in a copy.

The one deliberate exception is ``coordinaxs.api._custom_types``, which repeats
`CKey`/`CDict` because ``coordinaxs.api`` must not depend on ``coordinax``.
"""

__all__: tuple[str, ...] = (
    "Ang",
    "Len",
    "Spd",
    "OptUSys",
    "Shape",
    "HasShape",
    "CKey",
    "CDict",
    "CDictT",
    "Ks",
    "Ds",
)

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable
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
# Array-related Types

Shape: TypeAlias = tuple[int, ...]


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover


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
CDictT = TypeVar("CDictT", bound=CDict)

Ks = TypeVar("Ks", bound=tuple[CKey, ...], default=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...], default=tuple[str | None, ...])
