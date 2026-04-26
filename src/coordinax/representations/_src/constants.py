"""Internal custom types for coordinax."""

__all__ = (
    "ANGLE",
    "AREA",
    "LENGTH",
    "SPEED",
    "ANGULAR_SPEED",
    "ACCELERATION",
    "ANGULAR_ACCELERATION",
    "TIME",
)

from typing import cast

import unxt as u

ANGLE: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("angle"))
LENGTH: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("length"))
AREA: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("area"))
SPEED: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("speed"))
ANGULAR_SPEED: u.AbstractDimension = cast(
    "u.AbstractDimension", u.dimension("angular speed")
)
ACCELERATION: u.AbstractDimension = cast(
    "u.AbstractDimension", u.dimension("acceleration")
)
ANGULAR_ACCELERATION: u.AbstractDimension = cast(
    "u.AbstractDimension", u.dimension("angular acceleration")
)
TIME: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("time"))
