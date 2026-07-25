"""Internal custom types for coordinax."""

__all__ = ("ANGLE", "LENGTH", "MAGNITUDE", "ONE", "RADIAN")

from typing import cast

import unxt as u

ANGLE = cast("u.AbstractDimension", u.dimension("angle"))
LENGTH: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("length"))
# Magnitude has no dedicated astropy physical type; it resolves to
# ``PhysicalType('unknown')``. Named here so the distance-modulus branches can
# reject known-but-unsupported dimensions (time, mass, ...) with a clear error.
MAGNITUDE: u.AbstractDimension = cast(
    "u.AbstractDimension", u.dimension_of(u.Q(1.0, "mag"))
)

ONE = u.unit("")
RADIAN = u.unit("radian")
