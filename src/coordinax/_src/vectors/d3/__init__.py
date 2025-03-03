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
    "Cartesian3D",
]

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .base_spherical import (
    AbstractSphericalAcc,
    AbstractSphericalPos,
    AbstractSphericalVel,
)
from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D
from .cylindrical import CylindricalAcc, CylindricalPos, CylindricalVel
from .generic import Cartesian3D
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
    register_convert,  # noqa: F401
    register_primitives,  # noqa: F401
    register_vectorapi,  # noqa: F401
)
