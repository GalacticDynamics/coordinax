"""N-dimensional."""

__all__ = [
    # Base
    "AbstractPosND",
    "AbstractVelND",
    "AbstractAccND",
    # Cartesian
    "CartesianPosND",
    "CartesianVelND",
    "CartesianAccND",
    # Poincare
    "PoincarePolarVector",
]

from .base import AbstractAccND, AbstractPosND, AbstractVelND
from .cartesian import CartesianAccND, CartesianPosND, CartesianVelND
from .poincare import PoincarePolarVector

# isort: split
from . import register_vconvert  # noqa: F401
