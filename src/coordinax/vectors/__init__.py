"""`coordinax.vectors` Module."""

__all__ = (
    "cconvert",
    "equivalent",
    "AbstractVector",
    "Point",
    "Coordinate",
    "Tangent",
    "ToUnitsOptions",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.vectors"):
    from ._src import (
        AbstractVector,
        Coordinate,
        Point,
        Tangent,
        ToUnitsOptions,
        equivalent,
    )
    from coordinaxs.api.representations import cconvert

del install_import_hook
