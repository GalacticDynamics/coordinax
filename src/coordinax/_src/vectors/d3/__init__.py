"""3-dimensional representations."""
# pylint: disable=duplicate-code

__all__ = [
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
]

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .base_spherical import (
    AbstractSphericalAcc,
    AbstractSphericalPos,
    AbstractSphericalVel,
)
from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D
from .cylindrical import CylindricalAcc, CylindricalPos, CylindricalVel
from .generic import CartesianGeneric3D
from .lonlatspherical import (
    LonCosLatSphericalVel,
    LonLatSphericalAcc,
    LonLatSphericalPos,
    LonLatSphericalVel,
)
from .mathspherical import MathSphericalAcc, MathSphericalPos, MathSphericalVel
from .spherical import SphericalAcc, SphericalPos, SphericalVel
from .spheroidal import ProlateSpheroidalAcc, ProlateSpheroidalPos, ProlateSpheroidalVel

# isort: split
from . import (
    constructor,  # noqa: F401
    transform,  # noqa: F401
)
