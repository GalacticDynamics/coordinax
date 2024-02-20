"""Carteisan vector."""

__all__ = [
    # Position
    "Cartesian1DVector",
    "RadialVector",
    # Differential
    "CartesianDifferential1D",
    "RadialDifferential",
]

from typing import ClassVar, final

import equinox as eqx

from vector._typing import BatchFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract1DVector, Abstract1DVectorDifferential

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


##############################################################################
# Velocity


@final
class CartesianDifferential1D(Abstract1DVectorDifferential):
    """Cartesian differential representation."""

    d_x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    """Differential d_x/d_<>."""

    vector_cls: ClassVar[type[Cartesian1DVector]] = Cartesian1DVector  # type: ignore[misc]


@final
class RadialDifferential(Abstract1DVectorDifferential):
    """Radial differential representation."""

    d_r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    """Differential d_r/d_<>."""

    vector_cls: ClassVar[type[RadialVector]] = RadialVector  # type: ignore[misc]
