# pylint: disable=duplicate-code
"""4-dimensional."""

from . import base, compat, spacetime
from .base import *
from .compat import *
from .spacetime import *

__all__: list[str] = []
__all__ += base.__all__
__all__ += spacetime.__all__
__all__ += compat.__all__
