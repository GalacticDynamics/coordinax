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
from . import (
    register_plum,  # noqa: F401
    register_primitives,  # noqa: F401
    register_unxt,  # noqa: F401
    register_vectorapi,  # noqa: F401
)
