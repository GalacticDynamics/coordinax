"""Carteisan vector."""

__all__ = [
    # Position
    "Cartesian1DVector",
    "RadialVector",
]

from typing import final

import equinox as eqx

from vector._typing import BatchFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract1DVector

##############################################################################
# Position


@final
class Cartesian1DVector(Abstract1DVector):
    """Cartesian vector representation."""

    x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    """x coordinate."""


@final
class RadialVector(Abstract1DVector):
    """Radial vector representation."""

    r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    """Radial coordinate."""
