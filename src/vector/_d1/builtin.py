"""Carteisan vector."""

__all__ = [
    # Position
    "Cartesian1DVector",
    "RadialVector",
]

import equinox as eqx

from vector._typing import BatchFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract1DVector

##############################################################################
# Position


class Cartesian1DVector(Abstract1DVector):
    """Cartesian vector representation."""

    x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    """x coordinate."""


class RadialVector(Abstract1DVector):
    """Radial vector representation."""

    r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    """Radial coordinate."""
