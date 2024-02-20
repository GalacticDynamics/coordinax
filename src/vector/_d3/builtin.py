"""Built-in vector classes."""

__all__ = [
    "Cartesian3DVector",
    "SphericalVector",
    "CylindricalVector",
]

from typing import final

import equinox as eqx

from vector._typing import BatchFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract3DVector

##############################################################################
# Position


@final
class Cartesian3DVector(Abstract3DVector):
    """Cartesian vector representation."""

    x: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    y: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    z: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


@final
class SphericalVector(Abstract3DVector):
    """Spherical vector representation."""

    r: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    theta: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)


@final
class CylindricalVector(Abstract3DVector):
    """Cylindrical vector representation."""

    rho: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
    z: BatchFloatScalarQ = eqx.field(converter=converter_quantity_array)
