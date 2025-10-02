"""Coordinax Operator package."""

__all__ = [
    # Functional API
    "operate",
    "simplify_op",
    # Classes
    "AbstractOperator",
    "AbstractCompositeOperator",
    "Pipe",
    "Identity",
    "Rotate",
    "Translate",
    "GalileanOp",
]

from .api import operate, simplify_op
from .base import AbstractOperator
from .composite import AbstractCompositeOperator
from .galilean import GalileanOp
from .identity import Identity
from .pipe import Pipe
from .rotate import Rotate
from .translate import Translate

# # isort: split
# from . import compat, register_simplify

# del register_simplify, compat
