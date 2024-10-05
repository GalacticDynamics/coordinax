"""Built-in vector classes."""

__all__ = [
    # Physics conventions
    "SphericalPosition",
    "SphericalVelocity",
    "SphericalAcceleration",
]

from functools import partial
from typing import final

import equinox as eqx

import quaxed.lax as qlax
import quaxed.numpy as jnp
from dataclassish.converters import Unless
from unxt import AbstractDistance, AbstractQuantity, Distance, Quantity

import coordinax._src.typing as ct
from .base_spherical import (
    AbstractSphericalAcceleration,
    AbstractSphericalPosition,
    AbstractSphericalVelocity,
    _180d,
    _360d,
)
from coordinax._src.checks import (
    check_azimuth_range,
    check_polar_range,
    check_r_non_negative,
)
from coordinax._src.converters import converter_azimuth_to_range
from coordinax._src.utils import classproperty

##############################################################################
# Position


# TODO: make this an alias for SphericalPolarPosition, the more correct description?
@final
class SphericalPosition(AbstractSphericalPosition):
    """Spherical-Polar coordinates.

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    r : Distance
        Radial distance r (slant distance to origin),
    theta : Quantity['angle']
        Polar angle [0, 180] [deg] where 0 is the z-axis.
    phi : Quantity['angle']
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.

    """

    r: ct.BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, partial(Distance.constructor, dtype=float))
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: ct.BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Inclination angle :math:`\theta \in [0,180]`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_r_non_negative(self.r)
        check_polar_range(self.theta)
        check_azimuth_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["SphericalVelocity"]:
        return SphericalVelocity


@SphericalPosition.constructor._f.register  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[SphericalPosition],
    *,
    r: AbstractQuantity,
    theta: AbstractQuantity,
    phi: AbstractQuantity,
) -> SphericalPosition:
    """Construct SphericalPosition, allowing for out-of-range values.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> cx.SphericalPosition.constructor(r=Quantity(3, "kpc"),
    ...                                 theta=Quantity(90, "deg"),
    ...                                 phi=Quantity(0, "deg"))
    SphericalPosition(
      r=Distance(value=f32[], unit=Unit("kpc")),
      theta=Quantity[...](value=f32[], unit=Unit("deg")),
      phi=Quantity[...](value=f32[], unit=Unit("deg"))
    )

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cx.SphericalPosition.constructor(r=Quantity(-3, "kpc"),
    ...                                        theta=Quantity(45, "deg"),
    ...                                        phi=Quantity(0, "deg"))
    >>> vec.r
    Distance(Array(3., dtype=float32), unit='kpc')
    >>> vec.theta
    Quantity['angle'](Array(135., dtype=float32), unit='deg')
    >>> vec.phi
    Quantity[...](Array(180., dtype=float32), unit='deg')

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cx.SphericalPosition.constructor(r=Quantity(3, "kpc"),
    ...                                        theta=Quantity(190, "deg"),
    ...                                        phi=Quantity(0, "deg"))
    >>> vec.r
    Distance(Array(3., dtype=float32), unit='kpc')
    >>> vec.theta
    Quantity['angle'](Array(170., dtype=float32), unit='deg')
    >>> vec.phi
    Quantity['angle'](Array(180., dtype=float32), unit='deg')

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cx.SphericalPosition.constructor(r=Quantity(3, "kpc"),
    ...                                        theta=Quantity(90, "deg"),
    ...                                        phi=Quantity(365, "deg"))
    >>> vec.phi
    Quantity['angle'](Array(5., dtype=float32), unit='deg')

    """
    # 1) Convert the inputs
    fields = SphericalPosition.__dataclass_fields__
    r = fields["r"].metadata["converter"](r)
    theta = fields["theta"].metadata["converter"](theta)
    phi = fields["phi"].metadata["converter"](phi)

    # 2) handle negative distances
    r_pred = r < jnp.zeros_like(r)
    r = qlax.select(r_pred, -r, r)
    phi = qlax.select(r_pred, phi + _180d, phi)
    theta = qlax.select(r_pred, _180d - theta, theta)

    # 3) Handle polar angle outside of [0, 180] degrees
    theta = jnp.mod(theta, _360d)  # wrap to [0, 360) deg
    theta_pred = theta < _180d
    theta = qlax.select(theta_pred, theta, _360d - theta)
    phi = qlax.select(theta_pred, phi, phi + _180d)

    # 4) Construct. This also handles the azimuthal angle wrapping
    return cls(r=r, theta=theta, phi=phi)


##############################################################################


@final
class SphericalVelocity(AbstractSphericalVelocity):
    """Spherical differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_theta: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[SphericalPosition]:
        return SphericalPosition

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["SphericalAcceleration"]:
        return SphericalAcceleration


##############################################################################


@final
class SphericalAcceleration(AbstractSphericalAcceleration):
    """Spherical differential representation."""

    d2_r: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    d2_theta: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Inclination acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[SphericalVelocity]:
        return SphericalVelocity
