"""Internal custom types for coordinax."""

__all__ = (
    "ANGLE",
    "AREA",
    "LENGTH",
)

from typing import cast

import unxt as u

ANGLE: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("angle"))
LENGTH: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("length"))
AREA: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("area"))
