"""coordinax: Coordinates in JAX."""
# pylint: disable=import-error

__all__ = (
    # modules
    "angle",
    "distance",
    "r",
    "vecs",
    "ops",
    "frames",
    # common distance objects
    "Distance",
    # common vecs objects
    "vector",
    "vconvert",
    "Vector",
    "KinematicSpace",
    # frame objects
    "Coordinate",
)

from .setup_package import install_import_hook

with install_import_hook("coordinax"):
    from . import angle, distance, frames, ops, r, vecs
    from ._version import version as __version__  # noqa: F401
    from .distance import Distance
    from .frames import Coordinate
    from .vecs import KinematicSpace, Vector, vconvert, vector

    # isort: split
    # Interoperability
    from . import _interop


# Cleanup
del _interop, install_import_hook
