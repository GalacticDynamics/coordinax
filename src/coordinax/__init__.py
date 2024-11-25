"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

coordinax: Vectors in JAX
"""
# pylint: disable=import-error

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax", RUNTIME_TYPECHECKER):
    from . import angle, distance, frames, operators
    from ._src import (
        d3,
        d4,
        dn,
        exceptions,
        funcs,
        space,
        typing,
        utils,
        vectors,
    )
    from ._src.d3 import *
    from ._src.d4 import *
    from ._src.dn import *
    from ._src.exceptions import *
    from ._src.funcs import *
    from ._src.space import *
    from ._src.typing import *
    from ._src.utils import *
    from ._src.vectors.base import *
    from ._src.vectors.d1 import *
    from ._src.vectors.d2 import *
    from ._version import version as __version__
    from .distance import Distance

    # isort: split
    # Register vector transformations
    from ._src import transform

    # isort: split
    # Interoperability
    from . import _interop
    from ._src import compat

__all__ = ["__version__", "operators", "distance", "angle", "frames", "Distance"]
__all__ += funcs.__all__
__all__ += vectors.base.__all__
__all__ += vectors.d1.__all__
__all__ += vectors.d2.__all__
__all__ += d3.__all__
__all__ += d4.__all__
__all__ += dn.__all__
__all__ += space.__all__
__all__ += exceptions.__all__
__all__ += typing.__all__
__all__ += utils.__all__


# Cleanup
del (
    vectors,
    space,
    exceptions,
    transform,
    typing,
    utils,
    d3,
    d4,
    dn,
    funcs,
    RUNTIME_TYPECHECKER,
    compat,
    _interop,
)
