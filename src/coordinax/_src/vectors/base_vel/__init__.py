"""Abstract Velocity."""

__all__ = ["AbstractVel", "VELOCITY_CLASSES"]

from .core import VELOCITY_CLASSES, AbstractVel

# Register by import
# isort: split
from . import register_primitives  # noqa: F401
