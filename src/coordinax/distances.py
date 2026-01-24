"""`coordinax.distance` module."""

__all__ = (
    # Classes
    "AbstractDistance",
    "Distance",
    "DistanceModulus",
    "Parallax",
)

from .setup_package import install_import_hook

with install_import_hook("coordinax.distance"):
    from ._src.distances import (
        AbstractDistance,
        Distance,
        DistanceModulus,
        Parallax,
    )


del install_import_hook
