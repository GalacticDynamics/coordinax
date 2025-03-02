"""`coordinax.distance` module."""

__all__ = [
    # Classes
    "AbstractDistance",
    "Distance",
    "DistanceModulus",
    "Parallax",
    # funcs
    "distance",
    "parallax",
    "distance_modulus",
]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.distance", RUNTIME_TYPECHECKER):
    from ._src.distances import (
        AbstractDistance,
        Distance,
        DistanceModulus,
        Parallax,
        distance,
        distance_modulus,
        parallax,
    )


del RUNTIME_TYPECHECKER, install_import_hook
