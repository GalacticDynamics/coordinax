"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

coordinax: Vectors in JAX
"""
# pylint: disable=import-error

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax", RUNTIME_TYPECHECKER):
    from . import angle, distance, frames, ops, vecs
    from ._version import version as __version__  # noqa: F401
    from .distance import Distance
    from .frames import Coordinate
    from .vecs import (
        CartesianPos3D,
        CartesianVel3D,
        FourVector,
        Space,
        SphericalPos,
        SphericalVel,
        vconvert,
        vector,
    )

    # isort: split
    # Interoperability
    from . import _interop

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
    "Space",
    # frame objects
    "Coordinate",
]


# Cleanup
del RUNTIME_TYPECHECKER, _interop
