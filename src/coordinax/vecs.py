"""`coordinax.vecs` Module."""
# ruff:noqa: F403

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.vecs", RUNTIME_TYPECHECKER):
    from ._src import vectors
    from ._src.vectors.api import normalize_vector, vconvert, vector
    from ._src.vectors.base import (
        AbstractVector,
        AttrFilter,
        ToUnitsOptions,
        VectorAttribute,
    )
    from ._src.vectors.base_acc import ACCELERATION_CLASSES, AbstractAcc
    from ._src.vectors.base_pos import POSITION_CLASSES, AbstractPos
    from ._src.vectors.base_vel import VELOCITY_CLASSES, AbstractVel
    from ._src.vectors.d1 import (
        AbstractAcc1D,
        AbstractPos1D,
        AbstractVel1D,
        CartesianAcc1D,
        CartesianPos1D,
        CartesianVel1D,
        RadialAcc,
        RadialPos,
        RadialVel,
    )
    from ._src.vectors.d2 import *
    from ._src.vectors.d3 import *
    from ._src.vectors.d4 import *
    from ._src.vectors.dn import *
    from ._src.vectors.exceptions import *
    from ._src.vectors.space import *

    # Register vector transformations, functions, etc.
    # isort: split
    from ._src.vectors import funcs, transform

    # Interoperability
    # isort: split
    from ._src.vectors import compat

__all__ = [
    # API
    "vector",
    "vconvert",
    "normalize_vector",
    # Base
    "AbstractVector",
    "AttrFilter",
    "VectorAttribute",
    "ToUnitsOptions",
    # Base Classes
    "AbstractPos",
    "AbstractVel",
    "AbstractAcc",
    # --- 1D ---
    # Base
    "AbstractPos1D",
    "AbstractVel1D",
    "AbstractAcc1D",
    # Radial
    "RadialPos",
    "RadialVel",
    "RadialAcc",
    # Cartesian
    "CartesianPos1D",
    "CartesianVel1D",
    "CartesianAcc1D",
    # --- Misc ---
    "POSITION_CLASSES",
    "VELOCITY_CLASSES",
    "ACCELERATION_CLASSES",
]
__all__ += vectors.d2.__all__
__all__ += vectors.d3.__all__
__all__ += vectors.d4.__all__
__all__ += vectors.dn.__all__
__all__ += vectors.space.__all__
__all__ += vectors.exceptions.__all__


del vectors, install_import_hook, RUNTIME_TYPECHECKER, compat, funcs, transform
