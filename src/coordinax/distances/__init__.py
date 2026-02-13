"""`coordinax.distance` module."""

__all__ = ("AbstractDistance", "Distance", "DistanceModulus", "Parallax")

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.distance"):
    from ._src import (
        AbstractDistance,
        Distance,
        DistanceModulus,
        Parallax,
    )


del setup_package
