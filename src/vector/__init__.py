# pylint: disable=import-error

"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

vector: Vectors in JAX
"""

from jaxtyping import install_import_hook

from ._version import version as __version__
from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("vector", RUNTIME_TYPECHECKER):
    from ._base import *
    from ._d1 import *
    from ._d2 import *
    from ._d3 import *
    from ._exceptions import *
    from ._transform import *
    from ._typing import *
    from ._utils import *

from . import _base, _d1, _d2, _d3, _exceptions, _transform, _typing, _utils

__all__ = ["__version__"]
__all__ += _base.__all__
__all__ += _d1.__all__
__all__ += _d2.__all__
__all__ += _d3.__all__
__all__ += _exceptions.__all__
__all__ += _transform.__all__
__all__ += _typing.__all__
__all__ += _utils.__all__


# Cleanup
del _base, _exceptions, _transform, _typing, _utils, _d1, _d2, _d3, RUNTIME_TYPECHECKER
