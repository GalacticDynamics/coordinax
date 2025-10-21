"""coordinax: Coordinates in JAX."""
# pylint: disable=import-error

__all__ = [
    # modules
    "angle",
    "distance",
    "vecs",
    "ops",
    "frames",
    # common distance objects
    "Distance",
    # common vecs objects
    "vector",
    "vconvert",
    "CartesianPos3D",
    "CartesianVel3D",
    "SphericalPos",
    "SphericalVel",
    "FourVector",
    "KinematicSpace",
    # frame objects
    "Coordinate",
]

from .setup_package import install_import_hook

with install_import_hook("coordinax"):
    from . import angle, distance, frames, ops, vecs
    from ._version import version as __version__  # noqa: F401
    from .distance import Distance
    from .frames import Coordinate
    from .vecs import (
        CartesianPos3D,
        CartesianVel3D,
        FourVector,
        KinematicSpace,
        SphericalPos,
        SphericalVel,
        vconvert,
        vector,
    )

    # isort: split
    # Interoperability
    from . import _interop


# Cleanup
del _interop, install_import_hook
