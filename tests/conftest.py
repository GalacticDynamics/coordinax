"""Test configuration for coordinax tests."""

__all__ = ("POSITION_CLASSES", "VELOCITY_CLASSES", "ACCELERATION_CLASSES")


from typing import Final

import coordinax as cx

POSITION_CLASSES: Final = (cx.roles.PhysDisp,)
VELOCITY_CLASSES: Final = (cx.roles.PhysVel,)
ACCELERATION_CLASSES: Final = (cx.roles.PhysAcc,)
