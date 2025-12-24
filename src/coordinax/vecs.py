"""`coordinax.vecs` Module."""

__all__ = (
    "AbstractVectorLike",
    "Vector",
    "KinematicSpace",
    "vconvert",
    "r",
)

from . import r
from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.vecs"):
    from ._src.vectors import AbstractVectorLike, KinematicSpace, Vector
    from coordinax_api import vconvert

del install_import_hook, RUNTIME_TYPECHECKER
