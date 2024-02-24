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

import array_api_jax_compat as xp
import equinox as eqx
from jax_quantity import Quantity
from jaxtyping import Shaped

from vector._typing import BatchableFloatScalarQ
from vector._utils import converter_quantity_array

from .base import Abstract3DVector, Abstract3DVectorDifferential

_0m = Quantity(0, "meter")
_0d = Quantity(0, "rad")
_pid = Quantity(xp.pi, "rad")
_2pid = Quantity(2 * xp.pi, "rad")

##############################################################################
# Position


@final
class Cartesian3DVector(Abstract3DVector):
    """Cartesian vector representation."""

    x: Shaped[Quantity["length"], "*#batch"] = eqx.field(
        converter=converter_quantity_array
    )
    r"""X-coordinate :math:`x \in [-\infty, \infty]."""

    y: Shaped[Quantity["length"], "*#batch"] = eqx.field(
        converter=converter_quantity_array
    )
    r"""Y-coordinate :math:`y \in [-\infty, \infty]."""

    z: Shaped[Quantity["length"], "*#batch"] = eqx.field(
        converter=converter_quantity_array
    )
    r"""Z-coordinate :math:`z \in [-\infty, \infty]."""


@final
class SphericalVector(Abstract3DVector):
    """Spherical vector representation."""

    r: Shaped[Quantity["length"], "*#batch"] = eqx.field(
        converter=converter_quantity_array
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        _ = eqx.error_if(
            self.r,
            xp.any(self.r < _0m),
            "Radial distance 'r' must be in the range [0, +inf).",
        )
        _ = eqx.error_if(
            self.theta,
            xp.any((self.theta < _0d) | (self.theta > _pid)),
            "Inclination 'theta' must be in the range [0, pi].",
        )
        _ = eqx.error_if(
            self.phi,
            xp.any((self.phi < _0d) | (self.phi >= _2pid)),
            "Azimuthal angle 'phi' must be in the range [0, 2 * pi).",
        )


@final
class CylindricalVector(Abstract3DVector):
    """Cylindrical vector representation."""

    rho: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    phi: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    z: BatchableFloatScalarQ = eqx.field(converter=converter_quantity_array)
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        _ = eqx.error_if(
            self.rho,
            xp.any(self.rho < _0m),
            "Cylindrical radial distance 'rho' must be in the range [0, +inf).",
        )
        _ = eqx.error_if(
            self.phi,
            xp.any((self.phi < _0d) | (self.phi >= _2pid)),
            "Azimuthal angle 'phi' must be in the range [0, 2 * pi).",
        )


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
