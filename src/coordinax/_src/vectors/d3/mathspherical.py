"""Built-in vector classes."""

__all__ = [
    "MathSphericalAcc",
    "MathSphericalPos",
    "MathSphericalVel",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx

from dataclassish.converters import Unless
from unxt.quantity import Quantity

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


@final
class MathSphericalPos(AbstractSphericalPos):
    """Spherical vector representation.

    .. note::

        This class follows the Mathematics conventions.

    Parameters
    ----------
    r : `coordinax.Distance`
        Radial distance r (slant distance to origin),
    theta : `coordinax.angle.Angle`
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    phi : `coordinax.angle.Angle`
        Polar angle [0, 180] [deg] where 0 is the z-axis.

    """

    r: BatchableDistance = eqx.field(converter=Unless(AbstractDistance, Distance.from_))
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: BatchableAngle = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Azimuthal angle, generally :math:`\theta \in [0,360)`."""

    phi: BatchableAngle = eqx.field(converter=Angle.from_)
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        checks.check_r_non_negative(self.r)
        checks.check_polar_range(self.phi)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["MathSphericalVel"]:  # type: ignore[override]
        return MathSphericalVel

    @override
    @partial(eqx.filter_jit)
    def norm(self) -> BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> s = cx.vecs.MathSphericalPos(r=u.Quantity(3, "km"),
        ...                              theta=u.Quantity(90, "deg"),
        ...                              phi=u.Quantity(0, "deg"))
        >>> s.norm()
        Distance(Array(3, dtype=int32, ...), unit='km')

        """
        return self.r


##############################################################################


@final
class MathSphericalVel(AbstractSphericalVel):
    """Spherical differential representation."""

    r: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    theta: ct.BatchableAngularSpeed = eqx.field(
        converter=Quantity["angular speed"].from_
    )
    r"""Azimuthal speed :math:`d\theta/dt \in [-\infty, \infty]."""

    phi: ct.BatchableAngularSpeed = eqx.field(converter=Quantity["angular speed"].from_)
    r"""Inclination speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[MathSphericalPos]:  # type: ignore[override]
        return MathSphericalPos

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["MathSphericalAcc"]:  # type: ignore[override]
        return MathSphericalAcc


##############################################################################


@final
class MathSphericalAcc(AbstractSphericalAcc):
    """Spherical acceleration representation."""

    r: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    theta: ct.BatchableAngularAcc = eqx.field(
        converter=Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    phi: ct.BatchableAngularAcc = eqx.field(
        converter=Quantity["angular acceleration"].from_
    )
    r"""Inclination acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[MathSphericalVel]:  # type: ignore[override]
        return MathSphericalVel
