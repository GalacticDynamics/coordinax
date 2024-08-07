# pylint: disable=import-error

"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

coordinax: Vectors in JAX
"""

from jaxtyping import install_import_hook

from . import operators
from ._coordinax import (
    _base,
    _base_acc,
    _base_pos,
    _base_vel,
    _d1,
    _d2,
    _d3,
    _d4,
    _dn,
    _exceptions,
    _funcs,
    _space,
    _transform,
    typing,
    utils,
)
from ._coordinax._base import *
from ._coordinax._base_acc import *
from ._coordinax._base_pos import *
from ._coordinax._base_vel import *
from ._coordinax._d1 import *
from ._coordinax._d2 import *
from ._coordinax._d3 import *
from ._coordinax._d4 import *
from ._coordinax._dn import *
from ._coordinax._exceptions import *
from ._coordinax._funcs import *
from ._coordinax._space import *
from ._coordinax._transform import *
from ._coordinax.typing import *
from ._coordinax.utils import *
from ._version import version as __version__
from .setup_package import RUNTIME_TYPECHECKER

__all__ = ["__version__", "operators"]
__all__ += _funcs.__all__
__all__ += _base.__all__
__all__ += _base_pos.__all__
__all__ += _base_vel.__all__
__all__ += _base_acc.__all__
__all__ += _d1.__all__
__all__ += _d2.__all__
__all__ += _d3.__all__
__all__ += _d4.__all__
__all__ += _dn.__all__
__all__ += _space.__all__
__all__ += _exceptions.__all__
__all__ += _transform.__all__
__all__ += typing.__all__
__all__ += utils.__all__

# Interoperability
# Astropy
from ._coordinax._interop import coordinax_interop_astropy  # noqa: E402

# Runtime Typechecker
install_import_hook("coordinax", RUNTIME_TYPECHECKER)

# Cleanup
del (
    _base,
    _base_vel,
    _base_pos,
    _base_acc,
    _space,
    _exceptions,
    _transform,
    typing,
    utils,
    _d1,
    _d2,
    _d3,
    _d4,
    _dn,
    _funcs,
    RUNTIME_TYPECHECKER,
    coordinax_interop_astropy,
)
