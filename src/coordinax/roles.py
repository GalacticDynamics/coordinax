"""`coordinax.roles` Module."""

__all__ = (
    "AbstractRole",
    "AbstractPhysicalRole",
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
    # Helpers
    "guess_role",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.roles"):
    from ._src.api import as_pos, guess_role
    from ._src.roles import (
        AbstractPhysicalRole,
        AbstractRole,
        PhysAcc,
        PhysVel,
        Point,
        PhysDisp,
        phys_acc,
        phys_disp,
        phys_vel,
        point,
    )


del install_import_hook, RUNTIME_TYPECHECKER
