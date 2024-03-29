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

from unxt import Distance, Quantity

import coordinax._typing as ct
from .base import Abstract3DVector, Abstract3DVectorDifferential
from coordinax._checks import check_phi_range, check_r_non_negative, check_theta_range
from coordinax._converters import converter_phi_to_range
from coordinax._utils import classproperty

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
    phi : Quantity['angle']
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    theta : Quantity['angle']
        Polar angle [0, 180] [deg] where 0 is the z-axis.

    """

    r: ct.BatchableDistance = eqx.field(
        converter=partial(Distance.constructor, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_phi_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    theta: ct.BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_r_non_negative(self.r)
        check_theta_range(self.theta)
        check_phi_range(self.phi)

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
        >>> s = SphericalVector(r=Quantity(3, "kpc"), phi=Quantity(0, "deg"),
        ...                     theta=Quantity(90, "deg"))
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
        converter=partial(Distance.constructor, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_phi_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_r_non_negative(self.r)
        check_theta_range(self.phi)
        check_phi_range(self.theta)

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


@final
class LonLatSphericalVector(AbstractSphericalVector):
    """Spherical vector representation.

    .. note::

        This class follows the Geographic / Astronomical convention.

    Parameters
    ----------
    distance : Distance
        Radial distance r (slant distance to origin),
    lon : Quantity['angle']
        The longitude (azimuthal) angle [0, 360) [deg] where 0 is the x-axis.
    lat : Quantity['angle']
        The latitude (polar angle) [-90, 90] [deg] where 90 is the z-axis.

    """

    distance: ct.BatchableDistance = eqx.field(
        converter=partial(Distance.constructor, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    lon: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_phi_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Longitude angle :math:`\phi \in [0,360)`."""

    lat: ct.BatchableAngle = eqx.field(
        converter=lambda x: Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
    )
    r"""Latitude angle :math:`\phi \in [-90,90]`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_r_non_negative(self.distance)
        check_phi_range(self.lon)
        check_theta_range(self.lat)

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

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    d_theta: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

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

    d_distance: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_lon: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Longitude speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_lat: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Latitude speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalVector]:
        return LonLatSphericalVector


@final
class LonCosLatSphericalDifferential(Abstract3DVectorDifferential):
    """Spherical differential representation."""

    d_distance: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_lon_coslat: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Longitude * cos(Latitude) speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_lat: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Latitude speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[LonLatSphericalVector]:
        return LonLatSphericalVector
