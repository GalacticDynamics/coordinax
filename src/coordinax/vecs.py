"""`coordinax.vecs` Module."""

__all__ = [
    # API
    "vector",
    "vconvert",
    "normalize_vector",
    "cartesian_vector_type",
    "time_derivative_vector_type",
    "time_antiderivative_vector_type",
    "time_nth_derivative_vector_type",
    "IrreversibleDimensionChange",
    # Base
    "AbstractVectorLike",
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
    "Cartesian3D",
    # --- 4D ---
    "AbstractPos4D",
    "FourVector",
    # --- N-D ---
    "AbstractPosND",
    "AbstractVelND",
    "AbstractAccND",
    "CartesianPosND",
    "CartesianVelND",
    "CartesianAccND",
    "PoincarePolarVector",
    # --- Space ---
    "Space",
    # --- Misc ---
    "POSITION_CLASSES",
    "VELOCITY_CLASSES",
    "ACCELERATION_CLASSES",
]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.vecs", RUNTIME_TYPECHECKER):
    from ._src import vectors
    from ._src.vectors.api import (
        cartesian_vector_type,
        normalize_vector,
        time_antiderivative_vector_type,
        time_derivative_vector_type,
        time_nth_derivative_vector_type,
        vconvert,
        vector,
    )
    from ._src.vectors.base import (
        AbstractVector,
        AbstractVectorLike,
        AttrFilter,
        ToUnitsOptions,
        VectorAttribute,
    )
    from ._src.vectors.base_acc import ACCELERATION_CLASSES, AbstractAcc
    from ._src.vectors.base_pos import POSITION_CLASSES, AbstractPos
    from ._src.vectors.base_vel import VELOCITY_CLASSES, AbstractVel
    from ._src.vectors.collection import Space
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
        Cartesian3D,
        CartesianAcc3D,
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
    from ._src.vectors.d4 import AbstractPos4D, FourVector
    from ._src.vectors.dn import (
        AbstractAccND,
        AbstractPosND,
        AbstractVelND,
        CartesianAccND,
        CartesianPosND,
        CartesianVelND,
        PoincarePolarVector,
    )
    from ._src.vectors.exceptions import IrreversibleDimensionChange


del vectors, install_import_hook, RUNTIME_TYPECHECKER
