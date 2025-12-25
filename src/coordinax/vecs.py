"""`coordinax.vecs` Module."""

__all__ = (
    "AbstractVectorLike",
    "Vector",
    "KinematicSpace",
    "vector",
    "vconvert",
    "r",
)

from . import r
from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.vecs"):
    from ._src.vectors import AbstractVectorLike, KinematicSpace, Vector, vector
    from coordinax_api import vconvert

del install_import_hook, RUNTIME_TYPECHECKER
