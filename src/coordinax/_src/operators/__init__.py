"""Coordinax Operator package."""

__all__ = (
    # Functional API
    "eval_op",
    # Classes
    "AbstractOperator",
    "AbstractCompositeOperator",
    "Pipe",
    "Identity",
    "Rotate",
    # Role-specialized primitive operators
    "Translate",
    "Boost",
    "AccelShift",
    # Composite
    "GalileanOp",
)

from .accelshift import AccelShift
from .base import AbstractOperator, eval_op
from .boost import Boost
from .composite import AbstractCompositeOperator
from .galilean import GalileanOp
from .identity import Identity
from .pipe import Pipe
from .rotate import Rotate
from .translate import Translate
