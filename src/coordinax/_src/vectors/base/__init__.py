"""Bases."""

__all__ = [
    "AbstractVectorLike",
    "AbstractVector",
    "AbstractCartesian",
    # Type Guards
    "is_vectorlike",
    "is_vector",
    # Flags
    "AttrFilter",
    # Utils
    "VectorAttribute",
    "ToUnitsOptions",
]

from .attribute import VectorAttribute
from .base import AbstractVectorLike, is_vectorlike
from .cartesian import AbstractCartesian
from .flags import AttrFilter
from .register_unxt import ToUnitsOptions
from .vector import AbstractVector, is_vector

# Register by import
# isort: split
from . import (
    register_constructors,  # noqa: F401
    register_dataclassish,  # noqa: F401
    register_primitives,  # noqa: F401
    register_unxt,  # noqa: F401
)
