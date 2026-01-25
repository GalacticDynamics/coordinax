"""`coordinax.roles` Module."""

__all__ = (
    "AbstractRole",
    "AbstractPhysRole",
    "AbstractCoordRole",
    "Point",
    "point",
    "as_pos",  # For an object, reinterpret a Point as a Pos
    # Physical roles
    "PhysDisp",
    "phys_disp",
    "PhysVel",
    "phys_vel",
    "PhysAcc",
    "phys_acc",
    # Coordinate-basis tangent roles
    "CoordDisp",
    "coord_disp",
    "CoordVel",
    "coord_vel",
    "CoordAcc",
    "coord_acc",
    # Helpers
    "guess_role",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.roles"):
    from ._src.api import as_pos, guess_role
    from ._src.roles import (
        AbstractCoordRole,
        AbstractPhysRole,
        AbstractRole,
        CoordAcc,
        CoordDisp,
        CoordVel,
        PhysAcc,
        PhysDisp,
        PhysVel,
        Point,
        coord_acc,
        coord_disp,
        coord_vel,
        phys_acc,
        phys_disp,
        phys_vel,
        point,
    )


del install_import_hook, RUNTIME_TYPECHECKER
