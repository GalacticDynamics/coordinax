"""`coordinax.objs` Module."""

__all__ = (
    "AbstractVectorLike",
    "Vector",
    "PointedVector",
    "AbstractCoordinate",
    "Coordinate",
    "as_pos",
    "vconvert",
    "cdict",
    "ToUnitsOptions",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.objs"):
    from ._src.api import as_pos, cdict
    from ._src.objects import (
        AbstractCoordinate,
        AbstractVectorLike,
        Coordinate,
        PointedVector,
        ToUnitsOptions,
        Vector,
    )
    from coordinax_api import vconvert

del install_import_hook, RUNTIME_TYPECHECKER
