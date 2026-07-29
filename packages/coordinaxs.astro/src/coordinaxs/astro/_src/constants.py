"""Internal custom types for coordinax."""

__all__ = ("ANGLE", "LENGTH", "MAGNITUDE")

from typing import cast

import unxt as u

# The casts are load-bearing: `u.dimension` / `u.dimension_of` are typed as
# returning `object`, so without them the annotations below do not hold and
# anything returning one of these from a `-> u.AbstractDimension` function
# fails to type-check.
ANGLE: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("angle"))
LENGTH: u.AbstractDimension = cast("u.AbstractDimension", u.dimension("length"))
# Magnitude has no dedicated astropy physical type; it resolves to
# ``PhysicalType('unknown')``. Named here so the distance-modulus branches can
# reject known-but-unsupported dimensions (time, mass, ...) with a clear error.
MAGNITUDE: u.AbstractDimension = cast(
    "u.AbstractDimension", u.dimension_of(u.Q(1.0, "mag"))
)
