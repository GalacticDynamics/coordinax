"""Utilities for Space. Private module."""

__all__: list[str] = []

from collections.abc import Sequence
from typing import TypeAlias

from astropy.units import PhysicalType as Dimension

import quaxed.numpy as jnp
import unxt as u

DimensionLike: TypeAlias = Dimension | str
Shape: TypeAlias = tuple[int, ...]


def _get_dimension_name(dim: DimensionLike, /) -> str:
    return u.dimension(dim)._physical_type_list[0]  # noqa: SLF001


def can_broadcast_shapes(shapes: Sequence[Shape], /) -> bool:
    """Check if the shapes can be broadcasted together."""
    try:
        jnp.broadcast_shapes(*shapes)
    except ValueError:
        return False
    return True
