"""Built-in vector classes."""

__all__ = ["SphericalAcc", "SphericalPos", "SphericalVel"]

from typing import final

import equinox as eqx
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless
from unxt.quantity import AbstractQuantity

import coordinax._src.typing as ct
from .base_spherical import (
    AbstractSphericalAcc,
    AbstractSphericalPos,
    AbstractSphericalVel,
    _180d,
    _360d,
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
    def differential_cls(cls) -> type["SphericalVel"]:
        return SphericalVel


@dispatch  # type: ignore[misc]
def vector(
    cls: type[SphericalPos],
    *,
    r: AbstractQuantity,
    theta: AbstractQuantity,
    phi: AbstractQuantity,
) -> SphericalPos:
    """Construct SphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                             theta=u.Quantity(90, "deg"),
    ...                             phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [ 3 90  0]>

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(-3, "km"),
    ...                             theta=u.Quantity(45, "deg"),
    ...                             phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [  3 135 180]>

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                             theta=u.Quantity(190, "deg"),
    ...                             phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [  3 170 180]>

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base from_ does this):

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                             theta=u.Quantity(90, "deg"),
    ...                             phi=u.Quantity(365, "deg"))
    >>> vec.phi
    Angle(Array(5, dtype=int32, ...), unit='deg')

    """
    # 1) Convert the inputs
    fields = SphericalPos.__dataclass_fields__
    r = fields["r"].metadata["converter"](r)
    theta = fields["theta"].metadata["converter"](theta)
    phi = fields["phi"].metadata["converter"](phi)

    # 2) handle negative distances
    r_pred = r < jnp.zeros_like(r)
    r = jnp.where(r_pred, -r, r)
    phi = jnp.where(r_pred, phi + _180d, phi)
    theta = jnp.where(r_pred, _180d - theta, theta)

    # 3) Handle polar angle outside of [0, 180] degrees
    theta = jnp.mod(theta, _360d)  # wrap to [0, 360) deg
    theta_pred = theta < _180d
    theta = jnp.where(theta_pred, theta, _360d - theta)
    phi = jnp.where(theta_pred, phi, phi + _180d)

    # 4) Construct. This also handles the azimuthal angle wrapping
    return cls(r=r, theta=theta, phi=phi)


##############################################################################


@final
class SphericalVel(AbstractSphericalVel):
    """Spherical velocity."""

    d_r: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_theta: ct.BatchableAngularSpeed = eqx.field(
        converter=u.Quantity["angular speed"].from_
    )
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=u.Quantity["angular speed"].from_
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[SphericalPos]:
        return SphericalPos

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["SphericalAcc"]:
        return SphericalAcc


##############################################################################


@final
class SphericalAcc(AbstractSphericalAcc):
    """Spherical differential representation."""

    d2_r: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    d2_theta: ct.BatchableAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Inclination acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[SphericalVel]:
        return SphericalVel
