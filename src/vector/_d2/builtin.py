"""Built-in vector classes."""

__all__ = [
    # Position
    "Cartesian2DVector",
    "PolarVector",
    # "LnPolarVector",
    # "Log10PolarVector",
    # Differential
    "CartesianDifferential2D",
    "PolarDifferential",
]

from typing import ClassVar, final

import equinox as eqx

from vector._typing import BatchableFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract2DVector, Abstract2DVectorDifferential

# =============================================================================
# 2D


@final
class Cartesian2DVector(Abstract2DVector):
    """Cartesian vector representation."""

    x: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    y: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)


@final
class PolarVector(Abstract2DVector):
    """Polar vector representation.

    We use the symbol `phi` instead of `theta` to adhere to the ISO standard.
    """

    r: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)


# class LnPolarVector(Abstract2DVector):
#     """Log-polar vector representation."""

#     lnr: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
#     theta: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)


# class Log10PolarVector(Abstract2DVector):
#     """Log10-polar vector representation."""

#     log10r: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
#     theta: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)


##############################################################################


@final
class CartesianDifferential2D(Abstract2DVectorDifferential):
    """Cartesian differential representation."""

    d_x: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_y: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)

    vector_cls: ClassVar[type[Cartesian2DVector]] = Cartesian2DVector  # type: ignore[misc]


@final
class PolarDifferential(Abstract2DVectorDifferential):
    """Polar differential representation."""

    d_r: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)

    vector_cls: ClassVar[type[PolarVector]] = PolarVector  # type: ignore[misc]
