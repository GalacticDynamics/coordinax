"""Built-in vector classes."""

__all__ = ["SphericalAcc", "SphericalPos", "SphericalVel"]

from typing import final
from typing_extensions import override

import equinox as eqx

import unxt as u
from dataclassish.converters import Unless

import coordinax._src.typing as ct
from .base_spherical import (
    AbstractSphericalAcc,
    AbstractSphericalPos,
    AbstractSphericalVel,
)
from coordinax._src.angles import Angle, BatchableAngle
from coordinax._src.distances import AbstractDistance, BatchableDistance, Distance
from coordinax._src.utils import classproperty
from coordinax._src.vectors import checks
from coordinax._src.vectors.converters import converter_azimuth_to_range

##############################################################################
# Position


# TODO: make this an alias for SphericalPolarPos, the more correct description?
@final
class SphericalPos(AbstractSphericalPos):
    """Spherical-Polar coordinates.

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    r : `coordinax.Distance`
        Radial distance r (slant distance to origin),
    theta : `coordinax.angle.Angle`
        Polar angle [0, 180] [deg] where 0 is the z-axis.
    phi : `coordinax.angle.Angle`
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.

    """

    r: BatchableDistance = eqx.field(converter=Unless(AbstractDistance, Distance.from_))
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: BatchableAngle = eqx.field(converter=Angle.from_)
    r"""Inclination angle :math:`\theta \in [0,180]`."""

    phi: BatchableAngle = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Azimuthal angle, generally :math:`\phi \in [0,360)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        checks.check_r_non_negative(self.r)
        checks.check_polar_range(self.theta)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["SphericalVel"]:  # type: ignore[override]
        return SphericalVel


##############################################################################


@final
class SphericalVel(AbstractSphericalVel):
    """Spherical velocity."""

    r: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    theta: ct.BatchableAngularSpeed = eqx.field(
        converter=u.Quantity["angular speed"].from_
    )
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

    phi: ct.BatchableAngularSpeed = eqx.field(
        converter=u.Quantity["angular speed"].from_
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[SphericalPos]:  # type: ignore[override]
        return SphericalPos

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["SphericalAcc"]:  # type: ignore[override]
        return SphericalAcc


##############################################################################


@final
class SphericalAcc(AbstractSphericalAcc):
    """Spherical differential representation."""

    r: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    theta: ct.BatchableAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Inclination acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    phi: ct.BatchableAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[SphericalVel]:  # type: ignore[override]
        return SphericalVel
