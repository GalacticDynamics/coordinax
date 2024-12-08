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
    "GalileanRotation",
    "GalileanSpatialTranslation",
    "GalileanTranslation",
    # Misc
    "VelocityBoost",
]

from . import galilean
from .api import simplify_op
from .base import AbstractOperator
from .boost import VelocityBoost
from .composite import AbstractCompositeOperator
from .galilean.base import AbstractGalileanOperator
from .galilean.boost import GalileanBoost
from .galilean.composite import GalileanOperator
from .galilean.rotation import GalileanRotation
from .galilean.spatial_translation import GalileanSpatialTranslation
from .galilean.translation import GalileanTranslation
from .identity import Identity
from .pipe import Pipe

# isort: split
from . import compat, register_simplify

del register_simplify, compat
