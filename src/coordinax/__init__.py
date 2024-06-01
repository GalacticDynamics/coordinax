# pylint: disable=import-error

"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

coordinax: Vectors in JAX
"""

from jaxtyping import install_import_hook

from . import (
    _base,
    _base_pos,
    _base_vel,
    _d1,
    _d2,
    _d3,
    _d4,
    _dn,
    _exceptions,
    _transform,
    _typing,
    _utils,
    operators,
)
from ._base import *
from ._base_pos import *
from ._base_vel import *
from ._d1 import *
from ._d2 import *
from ._d3 import *
from ._d4 import *
from ._dn import *
from ._exceptions import *
from ._transform import *
from ._typing import *
from ._utils import *
from ._version import version as __version__
from .setup_package import RUNTIME_TYPECHECKER

__all__ = ["__version__", "operators"]
__all__ += _base.__all__
__all__ += _base_pos.__all__
__all__ += _base_vel.__all__
__all__ += _d1.__all__
__all__ += _d2.__all__
__all__ += _d3.__all__
__all__ += _d4.__all__
__all__ += _dn.__all__
__all__ += _exceptions.__all__
__all__ += _transform.__all__
__all__ += _typing.__all__
__all__ += _utils.__all__


# Runtime Typechecker
install_import_hook("coordinax", RUNTIME_TYPECHECKER)

# Cleanup
del (
    _base,
    _base_vel,
    _base_pos,
    _exceptions,
    _transform,
    _typing,
    _utils,
    _d1,
    _d2,
    _d3,
    _d4,
    _dn,
    RUNTIME_TYPECHECKER,
)
