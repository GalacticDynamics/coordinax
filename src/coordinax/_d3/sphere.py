"""Built-in vector classes."""

__all__ = [
    "AbstractSphericalVector",
    "AbstractSphericalDifferential",
    # Physics conventions
    "SphericalVector",
    "SphericalDifferential",
    # Mathematics conventions
    "MathSphericalVector",
    "MathSphericalDifferential",
    # Geographic / Astronomical conventions
    "LonLatSphericalVector",
    "LonLatSphericalDifferential",
    "LonCosLatSphericalDifferential",
]

from abc import abstractmethod
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.array_api as xp
import quaxed.lax as qlax
from unxt import AbstractDistance, Distance, Quantity

import coordinax._typing as ct
from .base import Abstract3DVector, Abstract3DVectorDifferential
from coordinax._checks import (
    check_azimuth_range,
    check_polar_range,
    check_r_non_negative,
)
from coordinax._converters import converter_azimuth_to_range
from coordinax._utils import classproperty

_90d = Quantity(90, "deg")
_180d = Quantity(180, "deg")

##############################################################################
# Position


class AbstractSphericalVector(Abstract3DVector):
    """Abstract spherical vector representation."""

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractSphericalDifferential"]: ...


@final
class SphericalVector(AbstractSphericalVector):
    """Spherical vector representation.

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
        converter=lambda x: x
        if isinstance(x, AbstractDistance)
        else Distance.constructor(x, dtype=float)
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
    def differential_cls(cls) -> type["SphericalDifferential"]:
        return SphericalDifferential

    @partial(jax.jit)
    def norm(self) -> ct.BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import SphericalVector
        >>> s = SphericalVector(r=Quantity(3, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> s.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.r


@final
class MathSphericalVector(AbstractSphericalVector):
    """Spherical vector representation.

    .. note::

        This class follows the Mathematics conventions.

    Parameters
    ----------
    r : Distance
        Radial distance r (slant distance to origin),
    theta : Quantity['angle']
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    phi : Quantity['angle']
        Polar angle [0, 180] [deg] where 0 is the z-axis.

    """

    r: ct.BatchableDistance = eqx.field(
        converter=lambda x: x
        if isinstance(x, AbstractDistance)
        else Distance.constructor(x, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\theta \in [0,360)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_r_non_negative(self.r)
        check_azimuth_range(self.theta)
        check_polar_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["MathSphericalDifferential"]:
        return MathSphericalDifferential

    @partial(jax.jit)
    def norm(self) -> ct.BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import MathSphericalVector
        >>> s = MathSphericalVector(r=Quantity(3, "kpc"), theta=Quantity(90, "deg"),
        ...                         phi=Quantity(0, "deg"))
        >>> s.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.r


# ============================================================================


@final
class LonLatSphericalVector(AbstractSphericalVector):
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

    >>> cx.LonLatSphericalVector(lon=Quantity(0, "deg"), lat=Quantity(0, "deg"),
    ...                          distance=Quantity(3, "kpc"))
    LonLatSphericalVector(
      lon=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      lat=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("kpc"))
    )

    The longitude and latitude angles are in the range [0, 360) and [-90, 90] degrees,
    and the radial distance is non-negative.
    When initializing, the longitude is wrapped to the [0, 360) degrees range.

    >>> vec = cx.LonLatSphericalVector(lon=Quantity(365, "deg"),
    ...                                lat=Quantity(90, "deg"),
    ...                                distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(5., dtype=float32), unit='deg')

    The latitude is not wrapped, but it is checked to be in the [-90, 90] degrees range.

    .. skip: next

    >>> try:
    ...     cx.LonLatSphericalVector(lon=Quantity(0, "deg"), lat=Quantity(100, "deg"),
    ...                              distance=Quantity(3, "kpc"))
    ... except Exception as e:
    ...     print(e)
    The inclination angle must be in the range [0, pi]...

    Likewise, the radial distance is checked to be non-negative.

    .. skip: next

    >>> try:
    ...     cx.LonLatSphericalVector(lon=Quantity(0, "deg"), lat=Quantity(0, "deg"),
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
        converter=lambda x: Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
    )
    r"""Latitude (polar) angle :math:`\in [-90,90]`."""

    distance: ct.BatchableDistance = eqx.field(
        converter=lambda x: x
        if isinstance(x, AbstractDistance)
        else Distance.constructor(x, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_azimuth_range(self.lon)
        check_polar_range(self.lat, -Quantity(90, "deg"), Quantity(90, "deg"))
        check_r_non_negative(self.distance)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["LonLatSphericalDifferential"]:
        return LonLatSphericalDifferential

    @partial(jax.jit)
    def norm(self) -> ct.BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import LonLatSphericalVector
        >>> s = LonLatSphericalVector(lon=Quantity(0, "deg"), lat=Quantity(90, "deg"),
        ...                           distance=Quantity(3, "kpc"))
        >>> s.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.distance


@LonLatSphericalVector.constructor._f.register  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[LonLatSphericalVector],
    *,
    lon: Quantity["angle"],
    lat: Quantity["angle"],
    distance: Distance,
) -> LonLatSphericalVector:
    """Construct LonLatSphericalVector, allowing for out-of-range values.

    Examples
    --------
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> cx.LonLatSphericalVector.constructor(lon=Quantity(0, "deg"),
    ...                                      lat=Quantity(0, "deg"),
    ...                                      distance=Quantity(3, "kpc"))
    LonLatSphericalVector(
      lon=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      lat=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("kpc"))
    )

    The distance can be negative, which wraps the longitude by 180 degrees and
    flips the latitude:

    >>> vec = cx.LonLatSphericalVector.constructor(lon=Quantity(0, "deg"),
    ...                                            lat=Quantity(45, "deg"),
    ...                                            distance=Quantity(-3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(180., dtype=float32), unit='deg')
    >>> vec.lat
    Quantity['angle'](Array(-45., dtype=float32), unit='deg')
    >>> vec.distance
    Distance(Array(3., dtype=float32), unit='kpc')

    The latitude can be outside the [-90, 90] deg range, causing the longitude
    to be shifted by 180 degrees:

    >>> vec = cx.LonLatSphericalVector.constructor(lon=Quantity(0, "deg"),
    ...                                            lat=Quantity(-100, "deg"),
    ...                                            distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(180., dtype=float32), unit='deg')
    >>> vec.lat
    Quantity['angle'](Array(-80., dtype=float32), unit='deg')
    >>> vec.distance
    Distance(Array(3., dtype=float32), unit='kpc')

    >>> vec = cx.LonLatSphericalVector.constructor(lon=Quantity(0, "deg"),
    ...                                            lat=Quantity(100, "deg"),
    ...                                            distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(180., dtype=float32), unit='deg')
    >>> vec.lat
    Quantity['angle'](Array(80., dtype=float32), unit='deg')
    >>> vec.distance
    Distance(Array(3., dtype=float32), unit='kpc')

    The longitude can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cx.LonLatSphericalVector.constructor(lon=Quantity(365, "deg"),
    ...                                            lat=Quantity(0, "deg"),
    ...                                            distance=Quantity(3, "kpc"))
    >>> vec.lon
    Quantity['angle'](Array(5., dtype=float32), unit='deg')

    """
    # 1) Convert the inputs
    fields = LonLatSphericalVector.__dataclass_fields__
    lon = fields["lon"].metadata["converter"](lon)
    lat = fields["lat"].metadata["converter"](lat)
    distance = fields["distance"].metadata["converter"](distance)

    # 2) handle negative distances
    distance_pred = distance < xp.zeros_like(distance)
    distance = qlax.select(distance_pred, -distance, distance)
    lon = qlax.select(distance_pred, lon + _180d, lon)
    lat = qlax.select(distance_pred, -lat, lat)

    # 3) Handle latitude outside of [-90, 90] degrees
    lat_pred = lat < -_90d
    lat = qlax.select(lat_pred, -_180d - lat, lat)
    lon = qlax.select(lat_pred, lon + _180d, lon)

    lat_pred = lat > _90d
    lat = qlax.select(lat_pred, _180d - lat, lat)
    lon = qlax.select(lat_pred, lon + _180d, lon)

    # 4) Construct. This also handles the longitude wrapping
    return cls(lon=lon, lat=lat, distance=distance)


##############################################################################


class AbstractSphericalDifferential(Abstract3DVectorDifferential):
    """Spherical differential representation."""

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[SphericalVector]: ...


@final
class SphericalDifferential(Abstract3DVectorDifferential):
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
    def integral_cls(cls) -> type[SphericalVector]:
        return SphericalVector


@final
class MathSphericalDifferential(Abstract3DVectorDifferential):
    """Spherical differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_theta: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Inclination speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[MathSphericalVector]:
        return MathSphericalVector


@final
class LonLatSphericalDifferential(Abstract3DVectorDifferential):
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
    def integral_cls(cls) -> type[LonLatSphericalVector]:
        return LonLatSphericalVector


@final
class LonCosLatSphericalDifferential(Abstract3DVectorDifferential):
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
    def integral_cls(cls) -> type[LonLatSphericalVector]:
        return LonLatSphericalVector
