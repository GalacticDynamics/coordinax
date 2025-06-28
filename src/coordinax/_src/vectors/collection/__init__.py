"""Space."""

__all__ = ["KinematicSpace"]

from .core import KinematicSpace

# Register by import
# isort: split
from . import (
    register_dataclassish,  # noqa: F401
    register_primitives,  # noqa: F401
    register_vectorapi,  # noqa: F401
)
