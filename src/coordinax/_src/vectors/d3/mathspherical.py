"""Built-in vector classes."""

__all__ = [
    "MathSphericalAcc",
    "MathSphericalPos",
    "MathSphericalVel",
]

import functools as ft
from typing import final
from typing_extensions import override

import equinox as eqx

import unxt as u
from dataclassish.converters import Unless

import coordinax._src.custom_types as ct
from .base_spherical import (
    AbstractSphericalAcc,
    AbstractSphericalPos,
    AbstractSphericalVel,
)
from coordinax._src.angles import Angle, BatchableAngle
from coordinax._src.distances import AbstractDistance, BatchableDistance, Distance
from coordinax._src.vectors import checks
from coordinax._src.vectors.converters import converter_azimuth_to_range


@final
class MathSphericalPos(AbstractSphericalPos):
    """Spherical vector representation.

    .. note::

        This class follows the Mathematics conventions.

    Parameters
    ----------
    r
        Radial distance r (slant distance to origin),
    theta
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    phi
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
    @ft.partial(eqx.filter_jit)
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


@final
class MathSphericalVel(AbstractSphericalVel):
    """Spherical differential representation."""

    r: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    theta: ct.BBtAngularSpeed = eqx.field(converter=u.Quantity["angular speed"].from_)
    r"""Azimuthal speed :math:`d\theta/dt \in [-\infty, \infty]."""

    phi: ct.BBtAngularSpeed = eqx.field(converter=u.Quantity["angular speed"].from_)
    r"""Inclination speed :math:`d\phi/dt \in [-\infty, \infty]."""


@final
class MathSphericalAcc(AbstractSphericalAcc):
    """Spherical acceleration representation."""

    r: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    theta: ct.BBtAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    phi: ct.BBtAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Inclination acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""
