"""Internal custom types for coordinax."""

__all__ = ("ANGLE", "LENGTH", "ONE", "RADIAN")

from typing import cast

import unxt as u

ANGLE = cast("u.AbstractDimension", u.dimension("angle"))
LENGTH: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("length"))

ONE = u.unit("")
RADIAN = u.unit("radian")
