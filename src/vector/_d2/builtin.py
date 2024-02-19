"""Built-in vector classes."""

__all__ = [
    "Cartesian2DVector",
    "PolarVector",
    # "LnPolarVector",
    # "Log10PolarVector",
]

from typing import final

import equinox as eqx

from vector._typing import BatchFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract2DVector

# =============================================================================
# 2D


@final
class Cartesian2DVector(Abstract2DVector):
    """Cartesian vector representation."""

    x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    y: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


@final
class PolarVector(Abstract2DVector):
    """Polar vector representation.

    We use the symbol `phi` instead of `theta` to adhere to the ISO standard.
    """

    r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


# class LnPolarVector(Abstract2DVector):
#     """Log-polar vector representation."""

#     lnr: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
#     theta: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


# class Log10PolarVector(Abstract2DVector):
#     """Log10-polar vector representation."""

#     log10r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
#     theta: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
