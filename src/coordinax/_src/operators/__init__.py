"""Coordinax Operator package."""

__all__ = (
    # Functional API
    "operate",
    "simplify",
    # Classes
    "AbstractOperator",
    "AbstractCompositeOperator",
    "Pipe",
    "Identity",
    "Rotate",
    "Add",
    "GalileanOp",
)

from .add import Add
from .api import operate, simplify
from .base import AbstractOperator
from .composite import AbstractCompositeOperator
from .galilean import GalileanOp
from .identity import Identity
from .pipe import Pipe
from .rotate import Rotate

# isort: split
from . import register_api  # noqa: F401
