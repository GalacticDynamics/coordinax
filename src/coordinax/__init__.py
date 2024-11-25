"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

coordinax: Vectors in JAX
"""
# pylint: disable=import-error

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax", RUNTIME_TYPECHECKER):
    from . import angle, distance, frames, operators
    from ._src import typing, utils, vectors
    from ._src.typing import *
    from ._src.utils import *
    from ._src.vectors.base import *
    from ._src.vectors.d1 import *
    from ._src.vectors.d2 import *
    from ._src.vectors.d3 import *
    from ._src.vectors.d4 import *
    from ._src.vectors.dn import *
    from ._src.vectors.exceptions import *
    from ._src.vectors.funcs import *
    from ._src.vectors.space import *
    from ._version import version as __version__
    from .distance import Distance

    # isort: split
    # Register vector transformations
    from ._src.vectors import transform

    # isort: split
    # Interoperability
    from . import _interop
    from ._src.vectors import compat

__all__ = ["__version__", "operators", "distance", "angle", "frames", "Distance"]
__all__ += vectors.funcs.__all__
__all__ += vectors.base.__all__
__all__ += vectors.d1.__all__
__all__ += vectors.d2.__all__
__all__ += vectors.d3.__all__
__all__ += vectors.d4.__all__
__all__ += vectors.dn.__all__
__all__ += vectors.space.__all__
__all__ += vectors.exceptions.__all__
__all__ += typing.__all__
__all__ += utils.__all__


# Cleanup
del vectors, transform, typing, utils, RUNTIME_TYPECHECKER, compat, _interop
