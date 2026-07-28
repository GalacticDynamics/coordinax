"""`coordinax.vectors` Module."""

__all__ = (
    "cconvert",
    "equivalent",
    "separation",
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
        equivalent,
        separation,
    )
    from coordinaxs.api.representations import cconvert

del install_import_hook
