"""Coordinax Operator package."""
# ruff: noqa: F401

__all__ = [
    "simplify_op",
    # Classes
    "AbstractOperator",
    "Identity",
    "AbstractCompositeOperator",
    "Pipe",
    # Galilean
    "AbstractGalileanOperator",
    "GalileanBoost",
    "GalileanOperator",
    # Misc
    "Rotate",
    "Translate",
    "VelocityBoost",
    # Utils
    "convert_to_pipe_operators",
]

from .api import simplify_op
from .base import AbstractOperator
from .boost import VelocityBoost
from .composite import AbstractCompositeOperator
from .galilean.base import AbstractGalileanOperator
from .galilean.boost import GalileanBoost
from .galilean.composite import GalileanOperator
from .galilean.spatial_translation import GalileanSpatialTranslation
from .identity import Identity
from .pipe import Pipe, convert_to_pipe_operators
from .rotate import Rotate
from .translate import Translate

# isort: split
from . import compat, register_simplify

del register_simplify, compat
