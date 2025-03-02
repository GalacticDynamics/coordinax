"""`coordinax.distance` module."""

__all__ = [
    # Classes
    "AbstractDistance",
    "Distance",
    "DistanceModulus",
    "Parallax",
    # funcs
    "parallax",
]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.distance", RUNTIME_TYPECHECKER):
    from ._src.distances import (
        AbstractDistance,
        Distance,
        DistanceModulus,
        Parallax,
        parallax,
    )


del RUNTIME_TYPECHECKER, install_import_hook
