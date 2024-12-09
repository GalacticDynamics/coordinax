"""Utilities for Space."""

__all__: list[str] = []

from typing import TypeAlias

from astropy.units import PhysicalType as Dimension

import quaxed.numpy as jnp
import unxt as u

DimensionLike: TypeAlias = Dimension | str


def _get_dimension_name(dim: DimensionLike, /) -> str:
    return u.dimension(dim)._physical_type_list[0]  # noqa: SLF001


def _can_broadcast_shapes(*shapes: tuple[int, ...]) -> bool:
    """Check if the shapes can be broadcasted together."""
    try:
        jnp.broadcast_shapes(*shapes)
    except ValueError:
        return False
    return True
