"""Custom types.

The single definition of ``coordinax``'s shared type vocabulary; every layer
re-exports from here.

``CKey`` and ``CDict`` are the exception: they live in
`coordinaxs.api.custom_types` and are re-exported below. ``coordinaxs.api`` is
the root of the workspace -- it may not depend on ``coordinax``, while every
other package depends on it -- so it is the only spot all layers can share.
Defining them here too would leave a copy that drifts silently: the
``dict``-not-``dict[...]`` trick below is load-bearing for plum's method cache.
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

from typing import Literal, Protocol, TypeAlias, runtime_checkable
from typing_extensions import TypeVar

import unxt as u

from coordinaxs.api.custom_types import CDict, CKey

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

CDictT = TypeVar("CDictT", bound=CDict)

Ks = TypeVar("Ks", bound=tuple[CKey, ...], default=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...], default=tuple[str | None, ...])
