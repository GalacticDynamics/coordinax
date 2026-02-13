"""`coordinax.objs` Module."""

__all__ = (
    "AbstractVectorLike",
    "Vector",
    "PointedVector",
    "AbstractCoordinate",
    "Coordinate",
    "as_disp",
    "vconvert",
    "cdict",
    "ToUnitsOptions",
)

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.objs"):
    from ._src import (
        AbstractCoordinate,
        AbstractVectorLike,
        Coordinate,
        PointedVector,
        ToUnitsOptions,
        Vector,
    )
    from coordinax.api import as_disp, cdict, vconvert

del setup_package
