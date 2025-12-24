"""Utilities for Space. Private module."""

__all__: tuple[str, ...] = ()


from collections.abc import Sequence

import quaxed.numpy as jnp
import unxt as u
from unxt._src.dimensions import name_of

from coordinax._src.custom_types import DimensionLike, Shape


def dimension_name(dim: DimensionLike, /) -> str:
    return name_of(u.dimension(dim))


def can_broadcast_shapes(shapes: Sequence[Shape], /) -> bool:
    """Check if the shapes can be broadcasted together."""
    try:
        jnp.broadcast_shapes(*shapes)
    except ValueError:
        return False
    else:
        return True
