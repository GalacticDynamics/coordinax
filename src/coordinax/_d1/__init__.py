# pylint: disable=duplicate-code
"""1-dimensional."""

from . import base, builtin, compat, transform
from .base import *
from .builtin import *
from .compat import *
from .transform import *

__all__: list[str] = []
__all__ += base.__all__
__all__ += builtin.__all__
__all__ += transform.__all__
__all__ += compat.__all__
