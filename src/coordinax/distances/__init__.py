"""`coordinax.distances` module."""

__all__ = ("AbstractDistance", "Distance", "DistanceModulus", "Parallax")

from ._setup_package import install_import_hook

with install_import_hook("coordinax.distance"):
    from ._src import (
        AbstractDistance,
        Distance,
        DistanceModulus,
        Parallax,
    )


del install_import_hook
