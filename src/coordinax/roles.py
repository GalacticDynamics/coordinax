"""`coordinax.roles` Module."""

__all__ = (
    "AbstractRole",
    "AbstractPhysicalRole",
    "Point",
    "point",
    "as_pos",  # For an object, reinterpret a Point as a Pos
    # Physical roles
    "Pos",
    "pos",
    "Vel",
    "vel",
    "Acc",
    "acc",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.roles"):
    from ._src.api import as_pos
    from ._src.roles import (
        AbstractPhysicalRole,
        AbstractRole,
        Acc,
        Point,
        Pos,
        Vel,
        acc,
        point,
        pos,
        vel,
    )


del install_import_hook, RUNTIME_TYPECHECKER
