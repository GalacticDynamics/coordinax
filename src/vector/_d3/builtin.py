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

from functools import partial
from typing import ClassVar, final

import array_api_jax_compat as xp
import equinox as eqx
import jax

from vector._checks import check_phi_range, check_r_non_negative, check_theta_range
from vector._typing import (
    BatchableAngle,
    BatchableAngularSpeed,
    BatchableLength,
    BatchableSpeed,
)
from vector._utils import converter_quantity_array

from .base import Abstract3DVector, Abstract3DVectorDifferential

##############################################################################
# Position


@final
class Cartesian3DVector(Abstract3DVector):
    """Cartesian vector representation."""

    x: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""Z coordinate :math:`z \in (-\infty,+\infty)`."""

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector."""
        return xp.sqrt(self.x**2 + self.y**2 + self.z**2)


@final
class SphericalVector(Abstract3DVector):
    """Spherical vector representation."""

    r: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: BatchableAngle = eqx.field(converter=converter_quantity_array)
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    phi: BatchableAngle = eqx.field(converter=converter_quantity_array)
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        check_r_non_negative(self.r)
        check_theta_range(self.theta)
        check_phi_range(self.phi)

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector."""
        return self.r


@final
class CylindricalVector(Abstract3DVector):
    """Cylindrical vector representation."""

    rho: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    phi: BatchableAngle = eqx.field(converter=converter_quantity_array)
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    z: BatchableLength = eqx.field(converter=converter_quantity_array)
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        check_r_non_negative(self.rho)
        check_phi_range(self.phi)

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector."""
        return xp.sqrt(self.rho**2 + self.z**2)


##############################################################################
# Differential


@final
class CartesianDifferential3D(Abstract3DVectorDifferential):
    """Cartesian differential representation."""

    d_x: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    d_y: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    d_z: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""Z speed :math:`dz/dt \in [-\infty, \infty]."""

    vector_cls: ClassVar[type[Cartesian3DVector]] = Cartesian3DVector  # type: ignore[misc]


@final
class SphericalDifferential(Abstract3DVectorDifferential):
    """Spherical differential representation."""

    d_r: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_theta: BatchableAngularSpeed = eqx.field(converter=converter_quantity_array)
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: BatchableAngularSpeed = eqx.field(converter=converter_quantity_array)
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    vector_cls: ClassVar[type[SphericalVector]] = SphericalVector  # type: ignore[misc]


@final
class CylindricalDifferential(Abstract3DVectorDifferential):
    """Cylindrical differential representation."""

    d_rho: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    d_phi: BatchableAngularSpeed = eqx.field(converter=converter_quantity_array)
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    d_z: BatchableSpeed = eqx.field(converter=converter_quantity_array)
    r"""Vertical speed :math:`dz/dt \in [-\infty, \infty]."""

    vector_cls: ClassVar[type[CylindricalVector]] = CylindricalVector  # type: ignore[misc]
