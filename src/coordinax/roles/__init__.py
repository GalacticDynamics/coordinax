"""`coordinax.roles` Module."""

__all__ = (
    "AbstractRole",
    "AbstractPhysRole",
    "AbstractCoordRole",
    "Point",
    "point",
    "as_disp",  # For an object, reinterpret a Point as a Pos
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

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.roles"):
    from ._src import (
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
    from coordinax.api import as_disp, guess_role


del setup_package
