"""Representation of coordinates in different systems."""

__all__ = ["ToUnitsOptions"]

from enum import Enum


class ToUnitsOptions(Enum):
    """Options for the units argument of `AbstractVector.uconvert`."""

    consistent = "consistent"
    """Convert to consistent units."""
