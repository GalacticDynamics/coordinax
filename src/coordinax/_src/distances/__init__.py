"""`coordinax.distance` private module.

Note that this module is private. Users should use the public API.

This module depends on the following modules:

- utils & typing

"""

__all__ = [
    "AbstractDistance",
    "Distance",
    "DistanceModulus",
    "Parallax",
    # Typing
    "BatchableLength",
    "BatchableDistance",
]

from .base import AbstractDistance
from .core import Distance, DistanceModulus, Parallax
from .typing import BatchableDistance, BatchableLength

# isort: split
# Register the dispatching
from . import compat, register_primitives  # noqa: F401
