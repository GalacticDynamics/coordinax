"""Coordinax Operator package."""

__all__ = (
    # Functional API
    "eval_op",
    # Classes
    "AbstractOperator",
    "AbstractCompositeOperator",
    "AbstractAdd",
    "Pipe",
    "Identity",
    "Rotate",
    # Role-specialized primitive operators
    "Translate",
    "Boost",
    # Composite
    "GalileanOp",
)

from .add import AbstractAdd
from .base import AbstractOperator, eval_op
from .boost import Boost
from .composite import AbstractCompositeOperator
from .galilean import GalileanOp
from .identity import Identity
from .pipe import Pipe
from .rotate import Rotate
from .translate import Translate

# isort: split
from . import register_apply  # noqa: F401