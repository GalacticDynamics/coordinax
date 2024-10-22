# pylint: disable=import-error

"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

coordinax: Vectors in JAX
"""

from jaxtyping import install_import_hook

from . import distance, operators
from ._src import (
    base,
    d1,
    d2,
    d3,
    d4,
    dn,
    exceptions,
    funcs,
    space,
    typing,
    utils,
)
from ._src.base import *
from ._src.d1 import *
from ._src.d2 import *
from ._src.d3 import *
from ._src.d4 import *
from ._src.dn import *
from ._src.exceptions import *
from ._src.funcs import *
from ._src.space import *
from ._src.typing import *
from ._src.utils import *
from ._version import version as __version__
from .distance import Distance
from .setup_package import RUNTIME_TYPECHECKER

__all__ = ["__version__", "operators", "distance", "Distance"]
__all__ += funcs.__all__
__all__ += base.__all__
__all__ += d1.__all__
__all__ += d2.__all__
__all__ += d3.__all__
__all__ += d4.__all__
__all__ += dn.__all__
__all__ += space.__all__
__all__ += exceptions.__all__
__all__ += typing.__all__
__all__ += utils.__all__

# isort: split
# Register vector transformations
from ._src import transform  # noqa: E402

# isort: split
# Interoperability
from . import _interop  # noqa: E402
from ._src import compat  # noqa: E402

# Runtime Typechecker
install_import_hook("coordinax", RUNTIME_TYPECHECKER)

# Cleanup
del (
    base,
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
