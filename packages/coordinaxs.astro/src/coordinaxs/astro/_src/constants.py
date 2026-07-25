"""Internal custom types for coordinax."""

__all__ = ("ANGLE", "LENGTH", "MAGNITUDE")

import unxt as u

ANGLE = u.dimension("angle")
LENGTH = u.dimension("length")
# Magnitude has no dedicated astropy physical type; it resolves to
# ``PhysicalType('unknown')``. Named here so the distance-modulus branches can
# reject known-but-unsupported dimensions (time, mass, ...) with a clear error.
MAGNITUDE = u.dimension_of(u.Q(1.0, "mag"))
