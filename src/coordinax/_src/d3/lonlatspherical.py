"""Built-in vector classes."""

__all__ = [
    "LonLatSphericalPosition",
    "LonLatSphericalVelocity",
    "LonLatSphericalAcceleration",
    "LonCosLatSphericalVelocity",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx

import quaxed.lax as qlax
import quaxed.numpy as jnp
from dataclassish.converters import Unless
from unxt import AbstractDistance, Distance, Quantity

import coordinax._src.typing as ct
from .base_spherical import (
    AbstractSphericalAcceleration,
    AbstractSphericalPosition,
    AbstractSphericalVelocity,
    _90d,
    _180d,
)
from coordinax._src.checks import (
    check_azimuth_range,
    check_polar_range,
    check_r_non_negative,
)
from coordinax._src.converters import converter_azimuth_to_range
from coordinax._src.utils import classproperty


@final
class LonLatSphericalPosition(AbstractSphericalPosition):
    """Spherical vector representation.

    .. note::

        This class follows the Geographic / Astronomical convention.

    Parameters
    ----------
    lon : Quantity['angle']
        The longitude (azimuthal) angle [0, 360) [deg] where 0 is the x-axis.
    lat : Quantity['angle']
        The latitude (polar angle) [-90, 90] [deg] where 90 is the z-axis.
    distance : Distance
        Radial distance r (slant distance to origin),

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.LonLatSphericalPosition(lon=Quantity(0, "deg"), lat=Quantity(0, "deg"),
    ...                          distance=Quantity(3, "kpc"))
    LonLatSphericalPosition(
      lon=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      lat=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("kpc"))
    )

    The longitude and latitude angles are in the range [0, 360) and [-90, 90] degrees,
    and the radial distance is non-negative.
    When initializing, the longitude is wrapped to the [0, 360) degrees range.

    >>> vec = cx.LonLatSphericalPosition(lon=Quantity(365, "deg"),
    ...                                lat=Quantity(90, "deg"),
    ...                                distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(5., dtype=float32), unit='deg')

    The latitude is not wrapped, but it is checked to be in the [-90, 90] degrees range.

    .. skip: next

    >>> try:
    ...     cx.LonLatSphericalPosition(lon=Quantity(0, "deg"), lat=Quantity(100, "deg"),
    ...                              distance=Quantity(3, "kpc"))
    ... except Exception as e:
    ...     print(e)
    The inclination angle must be in the range [0, pi]...

    Likewise, the radial distance is checked to be non-negative.

    .. skip: next

    >>> try:
    ...     cx.LonLatSphericalPosition(lon=Quantity(0, "deg"), lat=Quantity(0, "deg"),
    ...                              distance=Quantity(-3, "kpc"))
    ... except Exception as e:
    ...     print(e)
    The radial distance must be non-negative...

    """

    lon: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Longitude (azimuthal) angle :math:`\in [0,360)`."""

    lat: ct.BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Latitude (polar) angle :math:`\in [-90,90]`."""

    distance: ct.BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, partial(Distance.constructor, dtype=float))
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_azimuth_range(self.lon)
        check_polar_range(self.lat, -Quantity(90, "deg"), Quantity(90, "deg"))
        check_r_non_negative(self.distance)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["LonLatSphericalVelocity"]:
        return LonLatSphericalVelocity

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> ct.BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> s = cx.LonLatSphericalPosition(lon=Quantity(0, "deg"),
        ...                                lat=Quantity(90, "deg"),
        ...                                distance=Quantity(3, "kpc"))
        >>> s.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.distance


@LonLatSphericalPosition.constructor._f.register  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[LonLatSphericalPosition],
    *,
    lon: Quantity["angle"],
    lat: Quantity["angle"],
    distance: Distance,
) -> LonLatSphericalPosition:
    """Construct LonLatSphericalPosition, allowing for out-of-range values.

    Examples
    --------
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> cx.LonLatSphericalPosition.constructor(lon=Quantity(0, "deg"),
    ...                                      lat=Quantity(0, "deg"),
    ...                                      distance=Quantity(3, "kpc"))
    LonLatSphericalPosition(
      lon=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      lat=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("kpc"))
    )

    The distance can be negative, which wraps the longitude by 180 degrees and
    flips the latitude:

    >>> vec = cx.LonLatSphericalPosition.constructor(lon=Quantity(0, "deg"),
    ...                                              lat=Quantity(45, "deg"),
    ...                                              distance=Quantity(-3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(180., dtype=float32), unit='deg')
    >>> vec.lat
    Quantity['angle'](Array(-45., dtype=float32), unit='deg')
    >>> vec.distance
    Distance(Array(3., dtype=float32), unit='kpc')

    The latitude can be outside the [-90, 90] deg range, causing the longitude
    to be shifted by 180 degrees:

    >>> vec = cx.LonLatSphericalPosition.constructor(lon=Quantity(0, "deg"),
    ...                                              lat=Quantity(-100, "deg"),
    ...                                              distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(180., dtype=float32), unit='deg')
    >>> vec.lat
    Quantity['angle'](Array(-80., dtype=float32), unit='deg')
    >>> vec.distance
    Distance(Array(3., dtype=float32), unit='kpc')

    >>> vec = cx.LonLatSphericalPosition.constructor(lon=Quantity(0, "deg"),
    ...                                              lat=Quantity(100, "deg"),
    ...                                              distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(180., dtype=float32), unit='deg')
    >>> vec.lat
    Quantity['angle'](Array(80., dtype=float32), unit='deg')
    >>> vec.distance
    Distance(Array(3., dtype=float32), unit='kpc')

    The longitude can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cx.LonLatSphericalPosition.constructor(lon=Quantity(365, "deg"),
    ...                                              lat=Quantity(0, "deg"),
    ...                                              distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(5., dtype=float32), unit='deg')

    """
    # 1) Convert the inputs
    fields = LonLatSphericalPosition.__dataclass_fields__
    lon = fields["lon"].metadata["converter"](lon)
    lat = fields["lat"].metadata["converter"](lat)
    distance = fields["distance"].metadata["converter"](distance)

    # 2) handle negative distances
    distance_pred = distance < jnp.zeros_like(distance)
    distance = qlax.select(distance_pred, -distance, distance)
    lon = qlax.select(distance_pred, lon + _180d, lon)
    lat = qlax.select(distance_pred, -lat, lat)

    # 3) Handle latitude outside of [-90, 90] degrees
    # TODO: fix when lat < -180, lat > 180
    lat_pred = lat < -_90d
    lat = qlax.select(lat_pred, -_180d - lat, lat)
    lon = qlax.select(lat_pred, lon + _180d, lon)

    lat_pred = lat > _90d
    lat = qlax.select(lat_pred, _180d - lat, lat)
    lon = qlax.select(lat_pred, lon + _180d, lon)

    # 4) Construct. This also handles the longitude wrapping
    return cls(lon=lon, lat=lat, distance=distance)


##############################################################################


@final
class LonLatSphericalVelocity(AbstractSphericalVelocity):
    """Spherical differential representation."""

    d_lon: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Longitude speed :math:`dlon/dt \in [-\infty, \infty]."""

    d_lat: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Latitude speed :math:`dlat/dt \in [-\infty, \infty]."""

    d_distance: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalPosition]:
        return LonLatSphericalPosition

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["LonLatSphericalAcceleration"]:
        return LonLatSphericalAcceleration


@final
class LonCosLatSphericalVelocity(AbstractSphericalVelocity):
    """Spherical differential representation."""

    d_lon_coslat: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Longitude * cos(Latitude) speed :math:`dlon/dt \in [-\infty, \infty]."""

    d_lat: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Latitude speed :math:`dlat/dt \in [-\infty, \infty]."""

    d_distance: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalPosition]:
        return LonLatSphericalPosition

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["LonLatSphericalAcceleration"]:
        return LonLatSphericalAcceleration


##############################################################################


@final
class LonLatSphericalAcceleration(AbstractSphericalAcceleration):
    """Spherical acceleration representation."""

    d2_lon: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Longitude acceleration :math:`d^2lon/dt^2 \in [-\infty, \infty]."""

    d2_lat: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Latitude acceleration :math:`d^2lat/dt^2 \in [-\infty, \infty]."""

    d2_distance: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalVelocity]:
        return LonLatSphericalVelocity
