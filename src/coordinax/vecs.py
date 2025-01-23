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
    from ._src.vectors.d2 import (
        AbstractAcc2D,
        AbstractPos2D,
        AbstractVel2D,
        CartesianAcc2D,
        CartesianPos2D,
        CartesianVel2D,
        PolarAcc,
        PolarPos,
        PolarVel,
        TwoSphereAcc,
        TwoSpherePos,
        TwoSphereVel,
    )
    from ._src.vectors.d3 import (
        AbstractAcc3D,
        AbstractPos3D,
        AbstractSphericalAcc,
        AbstractSphericalPos,
        AbstractSphericalVel,
        AbstractVel3D,
        CartesianAcc3D,
        CartesianGeneric3D,
        CartesianPos3D,
        CartesianVel3D,
        CylindricalAcc,
        CylindricalPos,
        CylindricalVel,
        LonCosLatSphericalVel,
        LonLatSphericalAcc,
        LonLatSphericalPos,
        LonLatSphericalVel,
        MathSphericalAcc,
        MathSphericalPos,
        MathSphericalVel,
        ProlateSpheroidalAcc,
        ProlateSpheroidalPos,
        ProlateSpheroidalVel,
        SphericalAcc,
        SphericalPos,
        SphericalVel,
    )
    from ._src.vectors.d4 import *
    from ._src.vectors.dn import *
    from ._src.vectors.exceptions import *
    from ._src.vectors.space import *


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
    # --- 2D ---
    # Base
    "AbstractPos2D",
    "AbstractVel2D",
    "AbstractAcc2D",
    # TwoSphere
    "TwoSpherePos",
    "TwoSphereVel",
    "TwoSphereAcc",
    # Polar
    "PolarPos",
    "PolarVel",
    "PolarAcc",
    # Cartesian
    "CartesianPos2D",
    "CartesianVel2D",
    "CartesianAcc2D",
    # --- 3D ---
    # Base
    "AbstractPos3D",
    "AbstractVel3D",
    "AbstractAcc3D",
    # Cartesian
    "CartesianPos3D",
    "CartesianVel3D",
    "CartesianAcc3D",
    # Cylindrical
    "CylindricalPos",
    "CylindricalVel",
    "CylindricalAcc",
    # Base Spherical
    "AbstractSphericalPos",
    "AbstractSphericalVel",
    "AbstractSphericalAcc",
    # Spherical
    "SphericalPos",
    "SphericalVel",
    "SphericalAcc",
    # Math Spherical
    "MathSphericalPos",
    "MathSphericalVel",
    "MathSphericalAcc",
    # LonLat Spherical
    "LonLatSphericalPos",
    "LonLatSphericalVel",
    "LonLatSphericalAcc",
    "LonCosLatSphericalVel",
    # Prolate Spheroidal
    "ProlateSpheroidalPos",
    "ProlateSpheroidalVel",
    "ProlateSpheroidalAcc",
    # Generic
    "CartesianGeneric3D",
    # --- Misc ---
    "POSITION_CLASSES",
    "VELOCITY_CLASSES",
    "ACCELERATION_CLASSES",
]
__all__ += vectors.d4.__all__
__all__ += vectors.dn.__all__
__all__ += vectors.space.__all__
__all__ += vectors.exceptions.__all__


del vectors, install_import_hook, RUNTIME_TYPECHECKER
