# pylint: disable=duplicate-code
"""4-dimensional."""

from . import base, spacetime
from .base import *
from .spacetime import *

__all__: list[str] = []
__all__ += base.__all__
__all__ += spacetime.__all__
