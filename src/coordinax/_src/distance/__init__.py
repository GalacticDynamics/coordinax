"""Distance package."""

__all__ = ["AbstractDistance", "Distance", "DistanceModulus", "Parallax"]

from .base import AbstractDistance
from .core import Distance, DistanceModulus, Parallax

# isort: split
# Register the dispatching
from . import compat, register_primitives  # noqa: F401
