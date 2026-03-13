"""Intrinsic 2-sphere charts (non-Euclidean, no radial component)."""

__all__ = (
    "AbstractSphericalTwoSphere",
    "SphericalTwoSphere",
    "sph2",
    "LonLatSphericalTwoSphere",
    "lonlat_sph2",
    "LonCosLatSphericalTwoSphere",
    "loncoslat_sph2",
    "MathSphericalTwoSphere",
    "math_sph2",
)


import dataclasses

from typing import Final, Literal as L, final  # noqa: N817

import plum

import unxt as u

from . import checks
from .base import AbstractFixedComponentsChart, chart_dataclass_decorator
from .constants import Deg0, Deg90, Deg180
from .custom_types import CDict
from .d2 import Abstract2D, Cart2D
from .exceptions import NoGlobalCartesianChartError
from coordinax.internal.custom_types import Ang

_MSG_NO_CART2D: Final = (
    "{cls} has no global Cartesian 2D representation. Use an embedding "
    "via EmbeddedChart or a specific local projection if/when provided."
)


# ===============================================================================
# Base class for all spherical 2-sphere charts


class AbstractSphericalTwoSphere(Abstract2D):
    r"""Abstract base class for intrinsic charts on the unit two-sphere.

    All 2-sphere charts represent coordinates on the surface of a unit sphere.
    There is no global Cartesian 2D representation for these charts; they live
    on a curved 2D manifold.

    Concrete subclasses define specific coordinate systems:

    - Physics convention: polar (colatitude) and azimuthal angles
    - Longitude-latitude: geographic-style coordinates
    - Mathematics convention: azimuthal and polar angles (swapped)

    Notes
    -----
    - The intrinsic geometry is non-Euclidean (curved).
    - To embed in 3D Euclidean space, use an ``EmbeddedChart``.
    - ``cartesian_chart(...)`` raises ``NoGlobalCartesianChartError`` for all
      2-sphere charts.

    """


@plum.dispatch
def cartesian_chart(obj: AbstractSphericalTwoSphere, /) -> Cart2D:  # type: ignore[type-arg]
    """Raise NoGlobalCartesianChartError for any 2-sphere chart.

    2-sphere charts have no global Cartesian 2D representation.
    """
    raise NoGlobalCartesianChartError(_MSG_NO_CART2D.format(cls=type(obj).__name__))


# ===============================================================================
# SphericalTwoSphere  (theta, phi) — physics convention

SphericalTwoSphereKeys = tuple[L["theta"], L["phi"]]
SphericalTwoSphereDims = tuple[Ang, Ang]


@final
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class SphericalTwoSphere(
    AbstractFixedComponentsChart[SphericalTwoSphereKeys, SphericalTwoSphereDims],
    AbstractSphericalTwoSphere,
):
    r"""Intrinsic chart on the unit two-sphere with components ``(theta, phi)``.

    Uses the **physics convention**: $\theta \in [0,\pi]$ is the polar
    (colatitude) angle measured from the positive $z$-axis, and
    $\phi \in (-\pi,\pi]$ is the azimuthal angle.

    Parameters
    ----------
    theta
        Polar (colatitude) angle with angular units.
    phi
        Azimuthal angle with angular units.

    Notes
    -----
    - ``SphericalTwoSphere`` is a curved 2D manifold; there is no global Cartesian 2D
      chart. ``cartesian_chart(SphericalTwoSphere)`` raises.
    - The intrinsic metric is ``diag(1, sin^2 theta)``.
    - The longitude is undefined at the poles ``theta=0,pi``.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.sph2.components
    ('theta', 'phi')

    >>> cxc.sph2.coord_dimensions
    ('angle', 'angle')

    """

    def check_data(self, data: CDict, /) -> None:
        super(AbstractFixedComponentsChart, self).check_data(data)  # call base check
        checks.polar_range(u.Q.from_(data["theta"], "deg"), Deg0, Deg180)


sph2: Final = SphericalTwoSphere()


# ===============================================================================
# LonLatSphericalTwoSphere  (lon, lat)

LonLatSphericalTwoSphereKeys = tuple[L["lon"], L["lat"]]
LonLatSphericalTwoSphereDims = tuple[Ang, Ang]


@final
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class LonLatSphericalTwoSphere(
    AbstractFixedComponentsChart[
        LonLatSphericalTwoSphereKeys, LonLatSphericalTwoSphereDims
    ],
    AbstractSphericalTwoSphere,
):
    r"""Longitude-latitude chart on the two-sphere.

    Components are $(\mathrm{lon},\;\mathrm{lat})$ where:

    - ``lon`` is the azimuthal angle (longitude),
    - ``lat`` $\in [-\pi/2,\;\pi/2]$ is the latitude.

    Relation to :class:`SphericalTwoSphere`:

    $\mathrm{lat} = \pi/2 - \theta$, $\mathrm{lon} = \phi$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.lonlat_sph2.components
    ('lon', 'lat')

    >>> cxc.lonlat_sph2.coord_dimensions
    ('angle', 'angle')

    """

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)
        checks.polar_range(data["lat"], -Deg90, Deg90)


lonlat_sph2: Final = LonLatSphericalTwoSphere()


# ===============================================================================
# LonCosLatSphericalTwoSphere  (lon_coslat, lat)

LonCosLatSphericalTwoSphereKeys = tuple[L["lon_coslat"], L["lat"]]
LonCosLatSphericalTwoSphereDims = tuple[Ang, Ang]


@final
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class LonCosLatSphericalTwoSphere(
    AbstractFixedComponentsChart[
        LonCosLatSphericalTwoSphereKeys, LonCosLatSphericalTwoSphereDims
    ],
    AbstractSphericalTwoSphere,
):
    r"""Longitude-cos(latitude) chart on the two-sphere.

    Components are $(\mathrm{lon}\cos\mathrm{lat},\;\mathrm{lat})$.

    This form can improve numerical behavior near the poles because
    $\cos(\mathrm{lat}) \to 0$ as $|\mathrm{lat}| \to \pi/2$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.loncoslat_sph2.components
    ('lon_coslat', 'lat')

    >>> cxc.loncoslat_sph2.coord_dimensions
    ('angle', 'angle')

    """

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)
        checks.polar_range(data["lat"], -Deg90, Deg90)


loncoslat_sph2: Final = LonCosLatSphericalTwoSphere()


# ===============================================================================
# MathSphericalTwoSphere  (theta, phi) — mathematics convention

MathSphericalTwoSphereKeys = tuple[L["theta"], L["phi"]]


@final
@chart_dataclass_decorator
class MathSphericalTwoSphere(
    AbstractFixedComponentsChart[MathSphericalTwoSphereKeys, SphericalTwoSphereDims],
    AbstractSphericalTwoSphere,
):
    r"""Math-convention chart on the two-sphere.

    Components are $(\theta, \phi)$ where, contrary to the physics convention
    used in :class:`SphericalTwoSphere`:

    - $\theta$ is the **azimuthal** angle,
    - $\phi \in [0, \pi]$ is the **polar** angle.

    The conversion from :class:`SphericalTwoSphere` simply swaps the two angles.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.math_sph2.components
    ('theta', 'phi')

    >>> cxc.math_sph2.coord_dimensions
    ('angle', 'angle')

    """

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)
        checks.polar_range(data["phi"], Deg0, Deg180)


math_sph2: Final = MathSphericalTwoSphere()
