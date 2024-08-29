# pylint: disable=import-error

"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

coordinax: Vectors in JAX
"""

from jaxtyping import install_import_hook

from . import operators
from ._coordinax import (
    base,
    base_acc,
    base_pos,
    base_vel,
    d1,
    d2,
    d3,
    d4,
    dn,
    exceptions,
    funcs,
    space,
    transform,
    typing,
    utils,
)
from ._coordinax.base import *
from ._coordinax.base_acc import *
from ._coordinax.base_pos import *
from ._coordinax.base_vel import *
from ._coordinax.d1 import *
from ._coordinax.d2 import *
from ._coordinax.d3 import *
from ._coordinax.d4 import *
from ._coordinax.dn import *
from ._coordinax.exceptions import *
from ._coordinax.funcs import *
from ._coordinax.space import *
from ._coordinax.transform import *
from ._coordinax.typing import *
from ._coordinax.utils import *
from ._version import version as __version__
from .setup_package import RUNTIME_TYPECHECKER

__all__ = ["__version__", "operators"]
__all__ += funcs.__all__
__all__ += base.__all__
__all__ += base_pos.__all__
__all__ += base_vel.__all__
__all__ += base_acc.__all__
__all__ += d1.__all__
__all__ += d2.__all__
__all__ += d3.__all__
__all__ += d4.__all__
__all__ += dn.__all__
__all__ += space.__all__
__all__ += exceptions.__all__
__all__ += transform.__all__
__all__ += typing.__all__
__all__ += utils.__all__

# Interoperability
from . import _interop  # noqa: E402
from ._coordinax import compat  # noqa: E402

# Runtime Typechecker
install_import_hook("coordinax", RUNTIME_TYPECHECKER)

# Cleanup
del (
    base,
    base_vel,
    base_pos,
    base_acc,
    space,
    exceptions,
    transform,
    typing,
    utils,
    d1,
    d2,
    d3,
    d4,
    dn,
    funcs,
    RUNTIME_TYPECHECKER,
    compat,
    _interop,
)
