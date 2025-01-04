"""Acceleration Vector ABC.

This is Private API.

"""

__all__ = ["AbstractAcc", "ACCELERATION_CLASSES"]

from .core import ACCELERATION_CLASSES, AbstractAcc

# Register by import
# isort: split
from . import register_primitives  # noqa: F401
