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
    "BBtLength",
    "BatchableDistance",
]

from .base import AbstractDistance
from .custom_types import BatchableDistance, BBtLength
from .measures import Distance, DistanceModulus, Parallax

# isort: split
# Register the dispatching
from . import (  # noqa: F401
    register_constructors,
    register_converters,
    register_primitives,
)
