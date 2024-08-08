# pylint: disable=duplicate-code
"""N-dimensional."""

from . import base, cartesian, poincare, transform
from .base import *
from .cartesian import *
from .poincare import *
from .transform import *

__all__: list[str] = []
__all__ += base.__all__
__all__ += cartesian.__all__
__all__ += transform.__all__
__all__ += poincare.__all__
