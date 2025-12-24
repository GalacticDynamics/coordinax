"""Vector."""

__all__ = (
    # = 0D ======================================
    "Abstract0D",
    "Cart0D",
    "cart0d",
    # = 1D ======================================
    "Abstract1D",
    "Cart1D",
    "cart1d",
    "Radial1D",
    "radial1d",
    "Time1D",
    "time1d",
    # = 2D ======================================
    "Abstract2D",
    "Cart2D",
    "cart2d",
    "Polar2D",
    "polar2d",
    # = 3D ======================================
    "Abstract3D",
    "Cart3D",
    "cart3d",
    "Cylindrical3D",
    "cyl3d",
    "AbstractSpherical3D",
    "Spherical3D",
    "sph3d",
    "LonLatSpherical3D",
    "lonlatsph3d",
    "LonCosLatSpherical3D",
    "loncoslatsph3d",
    "MathSpherical3D",
    "mathsph3d",
    "ProlateSpheroidal3D",
    # = 6D ======================================
    "Abstract6D",
    "PoincarePolar6D",
    "poincarepolar6d",
    # = N-D =====================================
    "AbstractND",
    "CartND",
    "cartnd",
    "SpaceTimeEuclidean",
)

from dataclasses import KW_ONLY, dataclass, field

from collections.abc import Mapping
from jaxtyping import Float, Real
from typing import (
    Annotated,
    Any,
    Final,
    Literal as L,  # noqa: N817
    TypeVar,
    cast,
    final,
)
from typing_extensions import override

import plum
from beartype.vale import Is

import quaxed.numpy as jnp
import unxt as u
from dataclassish import replace

from . import checks
from .base import (
    AbstractChart,
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    AbstractFlatCartesianProductChart,
)
from coordinax._src import api
from coordinax._src.custom_types import Ang, CDict, Ds, Ks, Len, Spd

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # type: ignore[misc]
V = TypeVar("V")


_0d = u.Angle(0, "rad")
_pid = u.Angle(180, "deg")


def _is_not_abstract_chart_subclass(cls: type[Any], /) -> bool:
    """Check if cls is a non-abstract subclass of AbstractChart."""
    return not cls.__name__.startswith("Abstract") and not issubclass(
        cls, AbstractChart
    )


# fmt: off
# =========================================================
# 0D

class Abstract0D(AbstractDimensionalFlag, n=0):
    """Marker flag for 0D representations.

    A 0D representation has no coordinate component.
    """

    # TODO: add a check it's 0D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if _is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)

@plum.dispatch
def cartesian_chart(obj: Abstract0D, /) -> "Cart0D": return cart0d

# -----------------------------------------------

ZeroDKeys = tuple[()]
ZeroDDims = tuple[()]

@final
class Cart0D(AbstractFixedComponentsChart[ZeroDKeys, ZeroDDims], Abstract0D):
    """0D position representation."""

cart0d: Final = Cart0D()

# =========================================================
# 1D

class Abstract1D(AbstractDimensionalFlag, n=1):
    """Marker flag for 1D representations.

    A 1D representation has exactly one coordinate component. Examples include
    Cartesian $(x)$ or radial $(r)$ coordinates.
    """

    # TODO: add a check it's 1D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if _is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)

@plum.dispatch
def cartesian_chart(obj: Abstract1D, /) -> "Cart1D": return cart1d

# -----------------------------------------------
# Cartesian

Cart1DKeys = tuple[L["x"]]
Cart1DDims = tuple[Len]

@final
class Cart1D(AbstractFixedComponentsChart[Cart1DKeys, Cart1DDims], Abstract1D):
    pass

cart1d: Final = Cart1D()

# -----------------------------------------------
# Radial

RadialKeys = tuple[L["r"]]
Radial1DDims = tuple[Len]

@final
class Radial1D(AbstractFixedComponentsChart[RadialKeys, Radial1DDims], Abstract1D):
    pass

radial1d: Final = Radial1D()

# -----------------------------------------------
# Time

TimeKeys = tuple[L["t"]]
TimeDims = tuple[L["time"]]

@final
class Time1D(AbstractFixedComponentsChart[TimeKeys, TimeDims], Abstract1D):
    """1D time chart.

    A time chart has a single component "t" with time dimension.  This is the
    canonical 1D time chart used as the first factor in SpaceTimeCT product
    charts.

    """


time1d: Final = Time1D()

# Time1D is already Cartesian
@plum.dispatch
def cartesian_chart(obj: Time1D, /) -> Time1D:
    return time1d


# =========================================================
# 2D

class Abstract2D(AbstractDimensionalFlag, n=2):
    """Marker flag for 2D representations.

    A 2D representation has exactly two coordinate components. This does not
    imply that the underlying manifold is flat; for example, the two-sphere uses
    two angular coordinates but represents a curved surface.
    """

    # TODO: add a check it's 2D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if _is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)

@plum.dispatch
def cartesian_chart(obj: Abstract2D, /) -> "Cart2D": return cart2d

# -----------------------------------------------
# Cartesian

Cart2DKeys = tuple[L["x"], L["y"]]
Cart2DDims = tuple[Len, Len]

@final
class Cart2D(AbstractFixedComponentsChart[Cart2DKeys, Cart2DDims], Abstract2D):
    pass

cart2d: Final = Cart2D()

# -----------------------------------------------
# Polar

PolarKeys = tuple[L["r"], L["theta"]]
Polar2DDims = tuple[Len, Ang]

@final
class Polar2D(AbstractFixedComponentsChart[PolarKeys, Polar2DDims], Abstract2D):
    pass

polar2d: Final = Polar2D()

# =========================================================
# 3D

class Abstract3D(AbstractDimensionalFlag, n=3):
    """Marker flag for 3D representations.

    A 3D representation has exactly three coordinate components. These may
    parameterize Euclidean space (Cartesian), curvilinear coordinates
    (cylindrical, spherical), or other three-dimensional manifolds.
    """

    # TODO: add a check it's 3D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if _is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)


@plum.dispatch
def cartesian_chart(obj: Abstract3D, /) -> "Cart3D":
    return cart3d

# -----------------------------------------------
# Cartesian

Cart3DKeys = tuple[L["x"], L["y"], L["z"]]
Cart3DDims = tuple[Len, Len, Len]

@final
class Cart3D(AbstractFixedComponentsChart[Cart3DKeys, Cart3DDims], Abstract3D):
    pass

cart3d: Final = Cart3D()

# -----------------------------------------------
# Cylindrical

CylindricalKeys = tuple[L["rho"], L["phi"], L["z"]]
Cylindrical3DDims = tuple[Len, Ang, Len]

@final
class Cylindrical3D(
    AbstractFixedComponentsChart[CylindricalKeys, Cylindrical3DDims],
    Abstract3D
):
    pass

cyl3d: Final = Cylindrical3D()

# -----------------------------------------------
# Spherical

class AbstractSpherical3D(Abstract3D):
    """Abstract spherical vector representation."""

SphericalKeys = tuple[L["r"], L["theta"], L["phi"]]
Spherical3DDims = tuple[Len, Ang, Ang]

@final
class Spherical3D(
    AbstractFixedComponentsChart[SphericalKeys, Spherical3DDims],
    AbstractSpherical3D
):
    r"""Three-dimensional spherical coordinates $(r, \theta, \phi)$.

    The Cartesian embedding is

    $x = r\sin\theta\cos\phi,$
    $y = r\sin\theta\sin\phi,$
    $z = r\cos\theta.$

    Here $r \ge 0$, $\theta \in [0,\pi]$ is the polar (colatitude) angle,
    and $\phi$ is the azimuth.

    This convention matches the physics / mathematics definition of spherical
    coordinates.
    """

    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["theta"])

sph3d: Final = Spherical3D()

# -----------------------------------------------
# LonLatSpherical

LonLatSphericalKeys = tuple[L["lon"], L["lat"], L["distance"]]
LonLatSpherical3DDims = tuple[Ang, Ang, Len]

@final
class LonLatSpherical3D(
    AbstractFixedComponentsChart[LonLatSphericalKeys, LonLatSpherical3DDims],
    AbstractSpherical3D
):
    r"""Longitude-latitude spherical coordinates.

    Components are $(\mathrm{lon}, \mathrm{lat}, r)$ where:

    - ``lon`` is the azimuthal angle in the equatorial plane,
    - ``lat`` is the latitude in $[-\pi/2, \pi/2]$,
    - ``r`` is the radial distance.

    Relation to standard spherical coordinates:

    $\mathrm{lat} = \pi/2 - \theta$,  $\mathrm{lon} = \\phi$.
    """

    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["lat"], -u.Angle(90, "deg"), u.Angle(90, "deg"))

lonlatsph3d: Final = LonLatSpherical3D()

# -----------------------------------------------
# LonCosLatSpherical

LonCosLatSphericalKeys = tuple[L["lon_coslat"], L["lat"], L["distance"]]
LonCosLatSpherical3DDims = tuple[Ang, Ang, Len]

@final
class LonCosLatSpherical3D(
    AbstractFixedComponentsChart[LonCosLatSphericalKeys, LonCosLatSpherical3DDims],
    AbstractSpherical3D
):
    r"""Longitude-cos(latitude) spherical coordinates.

    Components are $(\mathrm{lon}\cos\mathrm{lat}, \mathrm{lat}, r)$.
    This form is sometimes used to improve numerical behavior near the poles,
    since $\cos(\mathrm{lat}) \to 0$ as $|\mathrm{lat}| \to \pi/2$.
    """

    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["lat"], -u.Angle(90, "deg"), u.Angle(90, "deg"))

loncoslatsph3d: Final = LonCosLatSpherical3D()

# -----------------------------------------------
# MathSpherical

MathSphericalKeys = tuple[L["r"], L["theta"], L["phi"]]

@final
class MathSpherical3D(
    AbstractFixedComponentsChart[MathSphericalKeys, Spherical3DDims],
    AbstractSpherical3D
):
    r"""Mathematical-convention spherical coordinates $(r, \theta, \phi)$.

    In this convention:

    - $\phi$ is the polar angle measured from the positive $z$ axis,
    - $\theta$ is the azimuthal angle in the $xy$-plane.

    This differs from the physics convention used by
    :class:`Spherical3D`, where $\theta$ and $\phi$ are swapped.
    """

    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["phi"], _0d, _pid)

mathsph3d: Final = MathSpherical3D()

# -----------------------------------------------
# Prolate Spheroidal

ProlateSpheroidalKeys = tuple[L["mu"], L["nu"], L["phi"]]
ProlateSpheroidal3DDims = tuple[L["area"], L["area"], Ang]

@final
@dataclass(frozen=True, slots=True)
class ProlateSpheroidal3D(
    AbstractFixedComponentsChart[ProlateSpheroidalKeys, ProlateSpheroidal3DDims],
    AbstractSpherical3D
):
    r"""Prolate spheroidal coordinates $(\mu, \nu, \phi)$ with focal length $\Delta$.

    These coordinates are adapted to systems with two foci separated by
    distance $2\Delta$.

    Validity constraints enforced by this representation:

    - $\Delta > 0$,
    - $\mu \ge \Delta^2$,
    - $|\nu| \le \Delta^2$.

    The parameter $\Delta$ is stored as metadata on the representation.
    """

    _: KW_ONLY
    Delta: Annotated[Real[u.quantity.StaticQuantity["length"], ""],
                     Is[lambda x: x.value > 0]]
    """Focal length of the coordinate system."""

    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.strictly_positive(self.Delta, name="Delta")
        checks.geq(
            data["mu"], self.Delta**2, name="mu", comp_name="Delta^2"
        )
        checks.leq(
            jnp.abs(data["nu"]), self.Delta**2, name="nu", comp_name="Delta^2"
        )

# =========================================================
# 6D

class Abstract6D(AbstractDimensionalFlag, n=6):
    """Marker flag for 6-D representations.

    An 6-D representation has an arbitrary number of coordinate components.
    Examples include Cartesian representations in arbitrary dimensions.
    """

    # TODO: add a check it's 6D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if _is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)
        # n is already fixed to 6
        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)



PoincarePolarKeys = tuple[
    L["rho"], L["pp_phi"], L["z"], L["dt_rho"], L["dt_pp_phi"], L["dt_z"]
]

PoincarePolarDims = tuple[
    Len, L["length / time**0.5"], Len, Spd, L["length / time**1.5"], Spd
]

@final
class PoincarePolar6D(
    AbstractFixedComponentsChart[PoincarePolarKeys, PoincarePolarDims],
    Abstract6D
):
    pass

poincarepolar6d: Final = PoincarePolar6D()

# =========================================================
# N-Dimensional

class AbstractND(AbstractDimensionalFlag, n="N"):
    """Marker flag for N-D representations.

    An N-D representation has an arbitrary number of coordinate components.
    Examples include Cartesian representations in arbitrary dimensions.
    """

    # TODO: add a check it's N-D

    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if _is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)
        # n is already fixed to "N"
        if n is not None:
            msg = f"{cls.__name__} does not support fixed n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)

@plum.dispatch(precedence=-1)
def cartesian_chart(obj: AbstractND, /) -> "CartND":
    return cartnd


# -----------------------------------------------
# Cartesian

CartNDKeys = tuple[L["q"]]
CartNDDims = tuple[Len]

@final
class CartND(AbstractFixedComponentsChart[CartNDKeys, CartNDDims], AbstractND):
    pass

cartnd: Final = CartND()


# =========================================================
# Special Charts

@final
@dataclass(frozen=True, slots=True)
class SpaceTimeEuclidean(AbstractFlatCartesianProductChart[Ks, Ds]):
    r"""4D Euclidean spacetime rep with components ``(ct, x, y, z)``.

    This is a Cartesian product chart: SpaceTimeCT(spatial_chart) ≡ time1d x
    spatial_chart

    The time component is always the canonical 1D time chart `time1d` with
    component "t". The time coordinate is automatically converted to ct using
    the speed of light.

    Mathematical definition:
    $$
       x^0 = ct,\quad x^i = \text{spatial components}
       \\
       g = \mathrm{diag}(1, 1, 1, 1)
    $$

    Parameters
    ----------
    spatial_chart
        Spatial position rep supplying component names and dimensions.
    c
        Speed of light used to form ``ct`` from ``t`` (defaults to
        ``Quantity(299_792.458, "km/s")``).

    Returns
    -------
    Rep
        Representation with components ``("ct", *spatial_chart.components)`` and
        dimensions ``("length", *spatial_chart.coord_dimensions)``.

    Notes
    -----
    - This is a rep (component schema), not stored numerical values.
    - The default metric is Euclidean (signature ``(+,+,+,+)``).
    - Use `coordinax.r.metric_of` to resolve the active metric.
    - The first factor is always `time1d`; the time chart is not
      user-selectable.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.charts.SpaceTimeEuclidean(cx.charts.cart3d)
    >>> p = {"ct": u.Q(1.0, "km"), "x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"),
    ...      "z": u.Q(0.0, "km")}
    >>> cx.metrics.metric_of(rep).metric_matrix(rep, p).shape
    (4, 4)

    """

    spatial_chart: AbstractChart[Any, Any] = field(default=cart3d)
    """Spatial part of the representation. Defaults: `coordinax.charts.cart3d`."""

    _: KW_ONLY
    c: Float[u.StaticQuantity["speed"], ""] = field(
        default=u.StaticQuantity(299_792.458, "km/s")
    )
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    @override
    @property
    def factors(self) -> tuple[AbstractChart[Any, Any], ...]:
        """Return (time1d, spatial_chart) as required by product chart spec."""
        return (time1d, self.spatial_chart)

    @override
    @property
    def factor_names(self) -> tuple[str, ...]:
        """Factor names are ('time', 'space')."""
        return ("time", "space")

    @property
    def components(self) -> Ks:
        # Override to add "ct" time component
        return cast("Ks", ("ct", *self.spatial_chart.components))

    @property
    def coord_dimensions(self) -> Ds:
        # Override to add "length" for ct dimension
        return cast("Ds", ("length", *self.spatial_chart.coord_dimensions))

    @override
    def split_components(self, p: CDict, /) -> tuple[CDict, ...]:  # type: ignore[override]
        """Split CsDict by factors, keeping 'ct' for time factor.

        SpaceTimeEuclidean uses 'ct' for the time component. The split returns
        factor dicts with their native keys ('ct' for time, spatial keys for space).
        """
        time = {"ct": p["ct"]}
        spatial = {k: p[k] for k in self.spatial_chart.components}
        return (time, spatial)

    @override
    def merge_components(self, parts: tuple[CDict, ...], /) -> CDict:  # type: ignore[override]
        """Merge factor CsDicts back into SpaceTimeEuclidean components.

        Expects time factor dict with 'ct' key, spatial factor dict with spatial keys.
        """
        return {**parts[0], **parts[1]}

    # def __hash__(self) -> int:
    #     # TODO: better hash, including more information
    #     return hash((self.__class__, self.spatial_chart.__class__))


@plum.dispatch
def cartesian_chart(obj: SpaceTimeEuclidean, /) -> SpaceTimeEuclidean: # type: ignore[type-arg]
    """Get a Cartesian-chart version of the given spacetime chart.

    Examples
    --------
    >>> import coordinax as cx
    >>> rep = cx.charts.SpaceTimeEuclidean(cx.charts.sph3d)
    >>> rep
    SpaceTimeEuclidean(
        spatial_chart=Spherical3D(), c=StaticQuantity(299792.458, 'km / s')
    )
    >>> cx.charts.cartesian_chart(rep)
    SpaceTimeEuclidean(
        spatial_chart=Cart3D(), c=StaticQuantity(299792.458, 'km / s')
    )

    """
    spatial_cart = api.cartesian_chart(obj.spatial_chart)
    # Return same object if already cartesian
    if spatial_cart == obj.spatial_chart:
        return obj
    return replace(obj, spatial_chart=spatial_cart)
