"""`coordinax.angle` private module.

Note that this module is private. Users should use the public API.

This module depends on the following modules:

- utils & typing

"""

__all__ = ["AbstractAngle", "Angle", "BatchableAngle", "BatchableAngleQ"]

from .base import AbstractAngle
from .core import Angle
from .typing import BatchableAngle, BatchableAngleQ

# isort: split
# Register the dispatching
from . import compat, register_primitives  # noqa: F401
