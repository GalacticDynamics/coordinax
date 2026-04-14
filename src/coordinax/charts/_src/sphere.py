"""Intrinsic 2-sphere charts (non-Euclidean, no radial component)."""

__all__ = (
    # Base
    "AbstractSphericalHyperSphere",
    # 1D
    "AbstractSphericalOneSphere",
    "CircularOneSphere",
    "sph1",
    # 2D
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


from typing import Any, Final, Literal as L, NoReturn, final  # noqa: N817
from typing_extensions import override

import jax.tree_util as jtu

import unxt as u

from . import checks
from .base import AbstractFixedComponentsChart, CDictT, chart_dataclass_decorator
from .constants import Deg0, Deg90, Deg180
from .d1 import Abstract1D
from .d2 import Abstract2D
from .exceptions import NoGlobalCartesianChartError
from coordinax.internal.custom_types import Ang, Ds, Ks

_MSG_NO_CART: Final = (
    "{cls} has no global Cartesian representation. Use an embedding "
    "via EmbeddedChart or a specific local projection if/when provided."
)


class AbstractSphericalHyperSphere(AbstractFixedComponentsChart[Ks, Ds]):
    r"""Abstract base class for intrinsic charts on the unit hypersphere.

    All hypersphere charts represent coordinates on the surface of a unit
    hypersphere.  There is no global Cartesian representation for these charts;
    they live on a curved manifold.

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

    @override
    @property
    def cartesian(self) -> NoReturn:
        """Raise NoGlobalCartesianChartError for any 2-sphere chart.

        2-sphere charts have no global Cartesian 2D representation.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> try:
        ...     cxc.sph2.cartesian
        ... except Exception as exc:
        ...     print(type(exc).__name__)
        ...     print("no global Cartesian representation" in str(exc))
        NoGlobalCartesianChartError
        True

        """
        raise NoGlobalCartesianChartError(_MSG_NO_CART.format(cls=type(self).__name__))


#####################################################################
# Base class for all circle charts


class AbstractSphericalOneSphere(AbstractSphericalHyperSphere[Ks, Ds], Abstract1D):
    r"""Abstract base class for intrinsic charts on the unit circle.

    All circle charts represent coordinates on the surface of a unit circle.
    There is no global Cartesian 1D representation for these charts; they live
    on a curved 1D manifold.

    Notes
    -----
    - The intrinsic geometry is non-Euclidean (curved).
    - To embed in higher Euclidean spaces, use an ``EmbeddedManifold``.
    - ``cartesian_chart(...)`` raises ``NoGlobalCartesianChartError`` for all
      circle charts.

    """


CircularOneSphereKeys = tuple[L["phi"]]  # TODO: theta?
CircularOneSphereDims = tuple[Ang]


@jtu.register_static
@final
@chart_dataclass_decorator
class CircularOneSphere(
    AbstractSphericalOneSphere[CircularOneSphereKeys, CircularOneSphereDims]
):
    """Standard circular coordinates on the unit circle.

    Parameters
    ----------
    phi
        Angular coordinate with angular units.

    Notes
    -----
    - ``CircularOneSphere`` is a curved 1D manifold; there is no global
      Cartesian 1D chart. ``cartesian_chart(CircularOneSphere)`` raises.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.sph1.components
    ('phi',)

    >>> cxc.sph1.coord_dimensions
    ('angle',)

    """


sph1: Final = CircularOneSphere()
"""Standard circular coordinates on the unit circle."""


#####################################################################
# Base class for all spherical 2-sphere charts


class AbstractSphericalTwoSphere(AbstractSphericalHyperSphere[Ks, Ds], Abstract2D):
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
    - To embed in 3D Euclidean space, use an ``EmbeddedManifold``.
    - ``cartesian_chart(...)`` raises ``NoGlobalCartesianChartError`` for all
      2-sphere charts.

    """


# ===============================================================================
# SphericalTwoSphere  (theta, phi) — physics convention


SphericalTwoSphereKeys = tuple[L["theta"], L["phi"]]
SphericalTwoSphereDims = tuple[Ang, Ang]


@jtu.register_static
@final
@chart_dataclass_decorator
class SphericalTwoSphere(
    AbstractSphericalTwoSphere[SphericalTwoSphereKeys, SphericalTwoSphereDims],
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

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        # call base check
        super().check_data(data, **kw)
        if values:
            checks.polar_range(u.Q.from_(data["theta"], "deg"), Deg0, Deg180)
        return data


sph2: Final = SphericalTwoSphere()
"""Standard spherical coordinates on the two-sphere with physics convention."""


# ===============================================================================
# LonLatSphericalTwoSphere  (lon, lat)

LonLatSphericalTwoSphereKeys = tuple[L["lon"], L["lat"]]
LonLatSphericalTwoSphereDims = tuple[Ang, Ang]


@jtu.register_static
@final
@chart_dataclass_decorator
class LonLatSphericalTwoSphere(
    AbstractSphericalTwoSphere[
        LonLatSphericalTwoSphereKeys, LonLatSphericalTwoSphereDims
    ]
):
    r"""Longitude-latitude chart on the two-sphere.

    Components are $(\mathrm{lon},\;\mathrm{lat})$ where:

    - ``lon`` is the azimuthal angle (longitude),
    - ``lat`` $\in [-\pi/2,\;\pi/2]$ is the latitude.

    Relation to {class}`SphericalTwoSphere`:

    $\mathrm{lat} = \pi/2 - \theta$, $\mathrm{lon} = \phi$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.lonlat_sph2.components
    ('lon', 'lat')

    >>> cxc.lonlat_sph2.coord_dimensions
    ('angle', 'angle')

    """

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)
        if values:
            checks.polar_range(data["lat"], -Deg90, Deg90)
        return data


lonlat_sph2: Final = LonLatSphericalTwoSphere()
"""Longitude-latitude spherical coordinates on the two-sphere."""


# ===============================================================================
# LonCosLatSphericalTwoSphere  (lon_coslat, lat)

LonCosLatSphericalTwoSphereKeys = tuple[L["lon_coslat"], L["lat"]]
LonCosLatSphericalTwoSphereDims = tuple[Ang, Ang]


@jtu.register_static
@final
@chart_dataclass_decorator
class LonCosLatSphericalTwoSphere(
    AbstractSphericalTwoSphere[
        LonCosLatSphericalTwoSphereKeys, LonCosLatSphericalTwoSphereDims
    ]
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

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)
        if values:
            checks.polar_range(data["lat"], -Deg90, Deg90)
        return data


loncoslat_sph2: Final = LonCosLatSphericalTwoSphere()
"""Longitude-cos(latitude) spherical coordinates on the two-sphere."""

# ===============================================================================
# MathSphericalTwoSphere  (theta, phi) — mathematics convention

MathSphericalTwoSphereKeys = tuple[L["theta"], L["phi"]]


@jtu.register_static
@final
@chart_dataclass_decorator
class MathSphericalTwoSphere(
    AbstractSphericalTwoSphere[MathSphericalTwoSphereKeys, SphericalTwoSphereDims]
):
    r"""Math-convention chart on the two-sphere.

    Components are $(\theta, \phi)$ where, contrary to the physics convention
    used in {class}`SphericalTwoSphere`:

    - $\theta$ is the **azimuthal** angle,
    - $\phi \in [0, \pi]$ is the **polar** angle.

    The conversion from {class}`SphericalTwoSphere` simply swaps the two angles.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.math_sph2.components
    ('theta', 'phi')

    >>> cxc.math_sph2.coord_dimensions
    ('angle', 'angle')

    """

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)
        if values:
            checks.polar_range(data["phi"], Deg0, Deg180)
        return data


math_sph2: Final = MathSphericalTwoSphere()
"""Standard spherical coordinates on the two-sphere with mathematics convention."""
