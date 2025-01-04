"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = ["spatial_component"]

from typing import Any

from plum import dispatch

from .base_pos import AbstractPos


@dispatch.abstract
def spatial_component(x: Any, /) -> Any:
    """Return the spatial component of the vector."""
    raise NotImplementedError  # pragma: no cover


@dispatch
def spatial_component(x: AbstractPos, /) -> AbstractPos:
    """Return the spatial component of the vector."""
    return x
