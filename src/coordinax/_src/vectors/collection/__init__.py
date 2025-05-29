"""Space."""

__all__ = ["AbstractVectors", "Space"]

from .base import AbstractVectors
from .core import Space

# Register by import
# isort: split
from . import (
    register_dataclassish,  # noqa: F401
    register_primitives,  # noqa: F401
    register_vectorapi,  # noqa: F401
)
