"""`coordinax.angle` private module.

Note that this module is private. Users should use the public API.

This module depends on the following modules:

- utils & typing

"""

__all__ = ["AbstractAngle", "Angle", "wrap_to", "BatchableAngle", "BatchableAngleQ"]

from .angle import Angle
from .base import AbstractAngle, wrap_to
from .custom_types import BatchableAngle, BatchableAngleQ

# isort: split
# Register the dispatching
from . import register_converters, register_primitives, register_unxt  # noqa: F401
