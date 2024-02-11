# pylint: disable=import-error

"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

vector: Vectors in JAX
"""

from . import _base, _builtin, _exceptions, _transform, _typing, _utils
from ._base import *
from ._builtin import *
from ._exceptions import *
from ._transform import *
from ._typing import *
from ._utils import *
from ._version import version as __version__

__all__ = ["__version__"]
__all__ += _base.__all__
__all__ += _builtin.__all__
__all__ += _exceptions.__all__
__all__ += _transform.__all__
__all__ += _typing.__all__
__all__ += _utils.__all__


# Cleanup
del _base, _builtin, _exceptions, _transform, _typing, _utils
