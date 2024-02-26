"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Iterator
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

import array_api_jax_compat as xp

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def dataclass_values(obj: "DataclassInstance") -> Iterator[Any]:
    """Return the values of a dataclass instance."""
    yield from (getattr(obj, f.name) for f in fields(obj))


def dataclass_items(obj: "DataclassInstance") -> Iterator[tuple[str, Any]]:
    """Return the field names and values of a dataclass instance."""
    yield from ((f.name, getattr(obj, f.name)) for f in fields(obj))


def full_shaped(obj: "AbstractVectorBase", /) -> "AbstractVectorBase":
    """Return the vector, fully broadcasting all components."""
    arrays = xp.broadcast_arrays(*dataclass_values(obj))
    return replace(obj, **dict(zip(obj.components, arrays, strict=True)))
