"""Carteisan vector."""

__all__ = [
    # 1D
    "Cartesian1DVector",
    "RadialVector",
    # 2D
    "Cartesian2DVector",
    "PolarVector",
    "LnPolarVector",
    "Log10PolarVector",
    # 3D
    "Cartesian3DVector",
    "SphericalVector",
    "CylindricalVector",
]

import equinox as eqx

from ._base import (
    Abstract1DVector,
    Abstract2DVector,
    Abstract3DVector,
)
from ._typing import BatchFloatScalarQ
from ._utils import converter_quantity_array

# =============================================================================
# 1D


class Cartesian1DVector(Abstract1DVector):
    """Cartesian vector representation."""

    x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


class RadialVector(Abstract1DVector):
    """Radial vector representation."""

    r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


# =============================================================================
# 2D


class Cartesian2DVector(Abstract2DVector):
    """Cartesian vector representation."""

    x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    y: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


class PolarVector(Abstract2DVector):
    """Polar vector representation.

    We use the symbol `phi` instead of `theta` to adhere to the ISO standard.
    """

    r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


class LnPolarVector(Abstract2DVector):
    """Log-polar vector representation."""

    lnr: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    theta: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


class Log10PolarVector(Abstract2DVector):
    """Log10-polar vector representation."""

    log10r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    theta: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


# =============================================================================
# 3D


class Cartesian3DVector(Abstract3DVector):
    """Cartesian vector representation."""

    x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    y: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    z: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


class SphericalVector(Abstract3DVector):
    """Spherical vector representation."""

    r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    theta: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


class CylindricalVector(Abstract3DVector):
    """Cylindrical vector representation."""

    rho: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    z: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
