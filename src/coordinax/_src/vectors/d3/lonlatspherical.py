"""Built-in vector classes."""

__all__ = [
    "LonCosLatSphericalVel",
    "LonLatSphericalAcc",
    "LonLatSphericalPos",
    "LonLatSphericalVel",
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
class LonLatSphericalPos(AbstractSphericalPos):
    """Spherical vector representation.

    .. note::

        This class follows the Geographic / Astronomical convention.

    Parameters
    ----------
    lon : `coordinax.angle.Angle`
        The longitude (azimuthal) angle [0, 360) [deg] where 0 is the x-axis.
    lat : `coordinax.angle.Angle`
        The latitude (polar angle) [-90, 90] [deg] where 90 is the z-axis.
    distance : Distance
        Radial distance r (slant distance to origin),

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Quantity(0, "deg"),
    ...                                  lat=u.Quantity(0, "deg"),
    ...                                  distance=u.Quantity(3, "km"))
    >>> print(vec)
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [0 0 3]>

    The longitude and latitude angles are in the range [0, 360) and [-90, 90] degrees,
    and the radial distance is non-negative.
    When initializing, the longitude is wrapped to the [0, 360) degrees range.

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Quantity(365, "deg"),
    ...                                  lat=u.Quantity(90, "deg"),
    ...                                  distance=u.Quantity(3, "km"))
    >>> vec.lon
    Angle(Array(5, dtype=int32, ...), unit='deg')

    The latitude is not wrapped, but it is checked to be in the [-90, 90] degrees range.

    .. skip: next

    >>> try:
    ...     cx.vecs.LonLatSphericalPos(lon=u.Quantity(0, "deg"),
    ...                                lat=u.Quantity(100, "deg"),
    ...                                distance=u.Quantity(3, "km"))
    ... except Exception as e:
    ...     print(e)
    The inclination angle must be in the range [0, pi]...

    Likewise, the radial distance is checked to be non-negative.

    .. skip: next

    >>> try:
    ...     cx.vecs.LonLatSphericalPos(lon=u.Quantity(0, "deg"),
    ...                                lat=u.Quantity(0, "deg"),
    ...                                distance=u.Quantity(-3, "km"))
    ... except Exception as e:
    ...     print(e)
    The radial distance must be non-negative...

    """

    lon: BatchableAngle = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Longitude (azimuthal) angle :math:`\in [0,360)`."""

    lat: BatchableAngle = eqx.field(converter=Angle.from_)
    r"""Latitude (polar) angle :math:`\in [-90,90]`."""

    distance: BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, Distance.from_)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        checks.check_polar_range(self.lat, -Angle(90, "deg"), Angle(90, "deg"))
        checks.check_r_non_negative(self.distance)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["LonLatSphericalVel"]:  # type: ignore[override]
        return LonLatSphericalVel

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> s = cx.vecs.LonLatSphericalPos(lon=u.Quantity(0, "deg"),
        ...                                lat=u.Quantity(90, "deg"),
        ...                                distance=u.Quantity(3, "km"))
        >>> s.norm()
        Distance(Array(3, dtype=int32, ...), unit='km')

        """
        return self.distance


##############################################################################


@final
class LonLatSphericalVel(AbstractSphericalVel):
    """Spherical velocity."""

    lon: ct.BatchableAngularSpeed = eqx.field(converter=Quantity["angular speed"].from_)
    r"""Longitude speed :math:`dlon/dt \in [-\infty, \infty]."""

    lat: ct.BatchableAngularSpeed = eqx.field(converter=Quantity["angular speed"].from_)
    r"""Latitude speed :math:`dlat/dt \in [-\infty, \infty]."""

    distance: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalPos]:  # type: ignore[override]
        return LonLatSphericalPos

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["LonLatSphericalAcc"]:  # type: ignore[override]
        return LonLatSphericalAcc


@final
class LonCosLatSphericalVel(AbstractSphericalVel):
    """Spherical differential representation."""

    lon_coslat: ct.BatchableAngularSpeed = eqx.field(
        converter=Quantity["angular speed"].from_
    )
    r"""Longitude * cos(Latitude) speed :math:`dlon/dt \in [-\infty, \infty]."""

    lat: ct.BatchableAngularSpeed = eqx.field(converter=Quantity["angular speed"].from_)
    r"""Latitude speed :math:`dlat/dt \in [-\infty, \infty]."""

    distance: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalPos]:  # type: ignore[override]
        return LonLatSphericalPos

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["LonLatSphericalAcc"]:  # type: ignore[override]
        return LonLatSphericalAcc


##############################################################################


@final
class LonLatSphericalAcc(AbstractSphericalAcc):
    """Spherical acceleration representation."""

    lon: ct.BatchableAngularAcc = eqx.field(
        converter=Quantity["angular acceleration"].from_
    )
    r"""Longitude acceleration :math:`d^2lon/dt^2 \in [-\infty, \infty]."""

    lat: ct.BatchableAngularAcc = eqx.field(
        converter=Quantity["angular acceleration"].from_
    )
    r"""Latitude acceleration :math:`d^2lat/dt^2 \in [-\infty, \infty]."""

    distance: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalVel]:  # type: ignore[override]
        return LonLatSphericalVel
