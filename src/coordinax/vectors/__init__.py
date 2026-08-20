"""`coordinax.vectors` Module."""

__all__ = (
    "cconvert",
    "equivalent",
    "chord_distance",
    "geodesic_distance",
    "AbstractVector",
    "Point",
    "Coordinate",
    "Tangent",
    "ToUnitsOptions",
)

from coordinax._src.setup_package import install_import_hook

with install_import_hook("coordinax.vectors"):
    from ._src import (
        AbstractVector,
        Coordinate,
        Point,
        Tangent,
        ToUnitsOptions,
        chord_distance,
        equivalent,
        geodesic_distance,
    )
    from coordinaxs.api.representations import cconvert

del install_import_hook
