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

from .setup_package import install_import_hook

with install_import_hook("coordinax.distance"):
    from ._src.distances import (
        AbstractDistance,
        Distance,
        DistanceModulus,
        Parallax,
        distance,
        distance_modulus,
        parallax,
    )


del install_import_hook
