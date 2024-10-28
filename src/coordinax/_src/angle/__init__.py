"""Angles package."""

__all__ = ["AbstractAngle", "Angle"]

from .base import AbstractAngle
from .core import Angle

# isort: split
# Register the dispatching
from . import compat, register_primitives  # noqa: F401
