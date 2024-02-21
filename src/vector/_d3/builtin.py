"""Built-in vector classes."""

__all__ = [
    # Position
    "Cartesian3DVector",
    "SphericalVector",
    "CylindricalVector",
    # Differential
    "CartesianDifferential3D",
    "SphericalDifferential",
    "CylindricalDifferential",
]

from typing import ClassVar, final

import equinox as eqx

from vector._typing import BatchableFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract3DVector, Abstract3DVectorDifferential

##############################################################################
# Position


@final
class Cartesian3DVector(Abstract3DVector):
    """Cartesian vector representation."""

    x: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    y: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    z: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)


@final
class SphericalVector(Abstract3DVector):
    """Spherical vector representation."""

    r: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    theta: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)


@final
class CylindricalVector(Abstract3DVector):
    """Cylindrical vector representation."""

    rho: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    z: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)


##############################################################################
# Differential


@final
class CartesianDifferential3D(Abstract3DVectorDifferential):
    """Cartesian differential representation."""

    d_x: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_y: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_z: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)

    vector_cls: ClassVar[type[Cartesian3DVector]] = Cartesian3DVector  # type: ignore[misc]


@final
class SphericalDifferential(Abstract3DVectorDifferential):
    """Spherical differential representation."""

    d_r: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_theta: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)

    vector_cls: ClassVar[type[SphericalVector]] = SphericalVector  # type: ignore[misc]


@final
class CylindricalDifferential(Abstract3DVectorDifferential):
    """Cylindrical differential representation."""

    d_rho: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    d_z: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)

    vector_cls: ClassVar[type[CylindricalVector]] = CylindricalVector  # type: ignore[misc]
