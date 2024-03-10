# pylint: disable=duplicate-code
"""3-dimensional representations."""

from . import base, builtin, compat, operate, transform
from .base import *
from .builtin import *
from .compat import *
from .operate import *
from .transform import *

__all__: list[str] = []
__all__ += base.__all__
__all__ += builtin.__all__
__all__ += transform.__all__
__all__ += operate.__all__
__all__ += compat.__all__
