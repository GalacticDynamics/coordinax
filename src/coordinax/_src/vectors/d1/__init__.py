# pylint: disable=duplicate-code
"""1-dimensional."""

__all__ = [
    # Base
    "AbstractPos1D",
    "AbstractVel1D",
    "AbstractAcc1D",
    # Radial
    "RadialPos",
    "RadialVel",
    "RadialAcc",
    # Cartesian
    "CartesianPos1D",
    "CartesianVel1D",
    "CartesianAcc1D",
]

from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from .cartesian import CartesianAcc1D, CartesianPos1D, CartesianVel1D
from .radial import RadialAcc, RadialPos, RadialVel

# isort: split
from . import (
    compat,  # noqa: F401
    transform,  # noqa: F401
)
