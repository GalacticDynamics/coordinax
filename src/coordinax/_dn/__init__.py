# pylint: disable=duplicate-code
"""N-dimensional."""

from . import base, builtin, poincare, transform
from .base import *
from .builtin import *
from .poincare import *
from .transform import *

__all__: list[str] = []
__all__ += base.__all__
__all__ += builtin.__all__
__all__ += transform.__all__
__all__ += poincare.__all__
