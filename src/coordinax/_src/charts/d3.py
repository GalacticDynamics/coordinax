"""3-Dimensional charts."""

__all__ = (
    "Abstract3D",
    "Cart3D",
    "cart3d",
    "Cylindrical3D",
    "cyl3d",
    "AbstractSpherical3D",
    "Spherical3D",
    "sph3d",
    "LonLatSpherical3D",
    "lonlat_sph3d",
    "LonCosLatSpherical3D",
    "loncoslat_sph3d",
    "MathSpherical3D",
    "math_sph3d",
    "ProlateSpheroidal3D",
)

import dataclasses

from jaxtyping import Real
from typing import Annotated, Any, Final, Literal as L, Self  # noqa: N817
from typing_extensions import override

import jax.tree_util as jtu
from beartype.vale import Is

import quaxed.numpy as jnp
import unxt as u

from coordinax._src.base import (
    MT,
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
    is_not_abstract_chart_subclass,
)
from coordinax._src.charts import checks
from coordinax._src.constants import Deg0, Deg90, Deg180
from coordinax._src.custom_types import Ang, CDictT, Ds, Ks, Len
from coordinax._src.euclidean.atlas import (
    EUCLIDEAN_ATLAS_DEFAULT_CHARTS,
    EuclideanAtlas,
)
from coordinax._src.euclidean.manifold import R3


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
        if is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)


# -----------------------------------------------
# Cartesian

Cart3DKeys = tuple[L["x"], L["y"], L["z"]]
Cart3DDims = tuple[Len, Len, Len]


@EuclideanAtlas.register
@jtu.register_static
@chart_dataclass_decorator
class Cart3D(AbstractFixedComponentsChart[MT, Cart3DKeys, Cart3DDims], Abstract3D):
    r"""Three-dimensional Cartesian chart $(x, y, z)$.

    Components are ordered as ``("x", "y", "z")`` with dimensions ``("length",
    "length", "length")``.

    This chart is the canonical 3D Cartesian chart and is returned by
    {func}`coordinax.charts.cartesian_chart` for 3D charts.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.cart3d.components
    ('x', 'y', 'z')

    >>> cxc.cart3d.coord_dimensions
    ('length', 'length', 'length')

    >>> isinstance(cxc.cartesian_chart(cxc.cart3d), cxc.Cart3D)
    True

    """

    _: dataclasses.KW_ONLY
    M: MT = R3  # ty: ignore[invalid-assignment]

    @override
    @property
    def cartesian(self) -> Self:
        """Return the canonical Cartesian chart for a 3D chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.cart3d.cartesian, cxc.Cart3D)
        True

        """
        return self


cart3d: Final = Cart3D(M=R3)
"""The canonical 3D Cartesian chart.

>>> import coordinax.charts as cxc
>>> cxc.cart3d.cartesian is cxc.cart3d
True

"""

EUCLIDEAN_ATLAS_DEFAULT_CHARTS[3] = cart3d


# -----------------------------------------------
# Cylindrical

CylindricalKeys = tuple[L["rho"], L["phi"], L["z"]]
Cylindrical3DDims = tuple[Len, Ang, Len]


@EuclideanAtlas.register
@jtu.register_static
@chart_dataclass_decorator
class Cylindrical3D(
    AbstractFixedComponentsChart[MT, CylindricalKeys, Cylindrical3DDims], Abstract3D
):
    r"""Three-dimensional cylindrical chart $(\rho, \phi, z)$.

    Components are ordered as ``("rho", "phi", "z")`` with dimensions
    ``("length", "angle", "length")``.

    This chart has direct transitions with {class}`coordinax.charts.Cart3D` and
    its canonical Cartesian projection is {obj}`coordinax.charts.cart3d`.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.cyl3d.components
    ('rho', 'phi', 'z')

    >>> cxc.cyl3d.coord_dimensions
    ('length', 'angle', 'length')

    >>> isinstance(cxc.cartesian_chart(cxc.cyl3d), cxc.Cart3D)
    True

    """

    _: dataclasses.KW_ONLY
    M: MT = R3  # ty: ignore[invalid-assignment]

    @override
    @property
    def cartesian(self) -> Cart3D[MT]:
        """Return the canonical Cartesian chart for a 3D cylindrical chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Cylindrical3D().cartesian, cxc.Cart3D)
        True
        """
        return Cart3D(M=self.M)


cyl3d: Final = Cylindrical3D(M=R3)
"""The canonical 3D cylindrical chart.

>>> import coordinax.charts as cxc
>>> cxc.cyl3d.cartesian is cxc.cart3d
True

"""

# -----------------------------------------------
# Spherical


class AbstractSpherical3D(AbstractFixedComponentsChart[MT, Ks, Ds], Abstract3D):
    """Abstract spherical vector representation."""

    _: dataclasses.KW_ONLY
    M: MT = R3  # ty: ignore[invalid-assignment]

    @override
    @property
    def cartesian(self) -> Cart3D[MT]:
        """Return the canonical Cartesian chart for a 3D chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Spherical3D().cartesian, cxc.Cart3D)
        True
        >>> isinstance(cxc.LonLatSpherical3D().cartesian, cxc.Cart3D)
        True
        >>> isinstance(cxc.MathSpherical3D().cartesian, cxc.Cart3D)
        True
        >>> isinstance(cxc.LonCosLatSpherical3D().cartesian, cxc.Cart3D)
        True

        """
        return Cart3D(M=self.M)


SphericalKeys = tuple[L["r"], L["theta"], L["phi"]]
Spherical3DDims = tuple[Len, Ang, Ang]


@EuclideanAtlas.register
@jtu.register_static
@chart_dataclass_decorator
class Spherical3D(AbstractSpherical3D[MT, SphericalKeys, Spherical3DDims]):
    r"""Three-dimensional spherical coordinates $(r, \theta, \phi)$.

    The Cartesian embedding is

    $x = r\sin\theta\cos\phi,$
    $y = r\sin\theta\sin\phi,$
    $z = r\cos\theta.$

    Here $r \ge 0$, $\theta \in [0,\pi]$ is the polar (colatitude) angle,
    and $\phi$ is the azimuth.

    This convention matches the physics / mathematics definition of spherical
    coordinates.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.sph3d.components
    ('r', 'theta', 'phi')

    >>> cxc.sph3d.coord_dimensions
    ('length', 'angle', 'angle')

    >>> isinstance(cxc.cartesian_chart(cxc.sph3d), cxc.Cart3D)
    True

    """

    _: dataclasses.KW_ONLY
    M: MT = R3  # ty: ignore[invalid-assignment]

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)
        if values:
            checks.polar_range(data["theta"])

        return data


sph3d: Final = Spherical3D(M=R3)
"""The canonical 3D spherical chart.

>>> import coordinax.charts as cxc
>>> cxc.sph3d.cartesian is cxc.cart3d
True

"""

# -----------------------------------------------
# LonLatSpherical

LonLatSphericalKeys = tuple[L["lon"], L["lat"], L["distance"]]
LonLatSpherical3DDims = tuple[Ang, Ang, Len]


@EuclideanAtlas.register
@jtu.register_static
@chart_dataclass_decorator
class LonLatSpherical3D(
    AbstractSpherical3D[MT, LonLatSphericalKeys, LonLatSpherical3DDims],
):
    r"""Longitude-latitude spherical coordinates.

    Components are $(\mathrm{lon}, \mathrm{lat}, r)$ where:

    - ``lon`` is the azimuthal angle in the equatorial plane,
    - ``lat`` is the latitude in $[-\pi/2, \pi/2]$,
    - ``r`` is the radial distance.

    Relation to standard spherical coordinates:

    $\mathrm{lat} = \pi/2 - \theta$,  $\mathrm{lon} = \\phi$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.lonlat_sph3d.components
    ('lon', 'lat', 'distance')

    >>> cxc.lonlat_sph3d.coord_dimensions
    ('angle', 'angle', 'length')

    >>> isinstance(cxc.cartesian_chart(cxc.lonlat_sph3d), cxc.Cart3D)
    True

    """

    _: dataclasses.KW_ONLY
    M: MT = R3  # ty: ignore[invalid-assignment]

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)  # call base check
        if values:
            checks.polar_range(data["lat"], -Deg90, Deg90)

        return data


lonlat_sph3d: Final = LonLatSpherical3D(M=R3)
"""The canonical longitude-latitude spherical chart.

>>> import coordinax.charts as cxc
>>> cxc.lonlat_sph3d.cartesian is cxc.cart3d
True

"""

# -----------------------------------------------
# LonCosLatSpherical

LonCosLatSphericalKeys = tuple[L["lon_coslat"], L["lat"], L["distance"]]
LonCosLatSpherical3DDims = tuple[Ang, Ang, Len]


@EuclideanAtlas.register
@jtu.register_static
@chart_dataclass_decorator
class LonCosLatSpherical3D(
    AbstractSpherical3D[MT, LonCosLatSphericalKeys, LonCosLatSpherical3DDims]
):
    r"""Longitude-cos(latitude) spherical coordinates.

    Components are $(\mathrm{lon}\cos\mathrm{lat}, \mathrm{lat}, r)$.
    This form is sometimes used to improve numerical behavior near the poles,
    since $\cos(\mathrm{lat}) \to 0$ as $|\mathrm{lat}| \to \pi/2$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.loncoslat_sph3d.components
    ('lon_coslat', 'lat', 'distance')

    >>> cxc.loncoslat_sph3d.coord_dimensions
    ('angle', 'angle', 'length')

    >>> isinstance(cxc.cartesian_chart(cxc.loncoslat_sph3d), cxc.Cart3D)
    True

    """

    _: dataclasses.KW_ONLY
    M: MT = R3  # ty: ignore[invalid-assignment]

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)  # call base check
        if values:
            checks.polar_range(data["lat"], -Deg90, Deg90)

        return data


loncoslat_sph3d: Final = LonCosLatSpherical3D(M=R3)
"""The canonical longitude-cos(latitude) spherical chart.

>>> import coordinax.charts as cxc
>>> cxc.loncoslat_sph3d.cartesian is cxc.cart3d
True

"""

# -----------------------------------------------
# MathSpherical

MathSphericalKeys = tuple[L["r"], L["theta"], L["phi"]]


@EuclideanAtlas.register
@jtu.register_static
@chart_dataclass_decorator
class MathSpherical3D(AbstractSpherical3D[MT, MathSphericalKeys, Spherical3DDims]):
    r"""Mathematical-convention spherical coordinates $(r, \theta, \phi)$.

    In this convention:

    - $\phi$ is the polar angle measured from the positive $z$ axis,
    - $\theta$ is the azimuthal angle in the $xy$-plane.

    This differs from the physics convention used by
    {class}`coordinax.charts.Spherical3D`, where $\theta$ and $\phi$ are
    swapped.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.math_sph3d.components
    ('r', 'theta', 'phi')

    >>> cxc.math_sph3d.coord_dimensions
    ('length', 'angle', 'angle')

    >>> isinstance(cxc.cartesian_chart(cxc.math_sph3d), cxc.Cart3D)
    True

    """

    _: dataclasses.KW_ONLY
    M: MT = R3  # ty: ignore[invalid-assignment]

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)  # call base check
        if values:
            checks.polar_range(data["phi"], Deg0, Deg180)
        return data


math_sph3d: Final = MathSpherical3D(M=R3)
"""The canonical mathematical-convention spherical chart.

>>> import coordinax.charts as cxc
>>> cxc.math_sph3d.cartesian is cxc.cart3d
True

"""

# -----------------------------------------------
# Prolate Spheroidal

ProlateSpheroidalKeys = tuple[L["mu"], L["nu"], L["phi"]]
ProlateSpheroidal3DDims = tuple[L["area"], L["area"], Ang]


@EuclideanAtlas.register
@jtu.register_static
@chart_dataclass_decorator
class ProlateSpheroidal3D(
    Abstract3D,
    AbstractFixedComponentsChart[MT, ProlateSpheroidalKeys, ProlateSpheroidal3DDims],
):
    r"""Prolate spheroidal coordinates $(\mu, \nu, \phi)$ with focal length $\Delta$.

    These coordinates are adapted to systems with two foci separated by
    distance $2\Delta$.

    Validity constraints enforced by this representation:

    - $\Delta > 0$,
    - $\mu \ge \Delta^2$,
    - $|\nu| \le \Delta^2$.

    The parameter $\Delta$ is stored as metadata on the representation.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> chart = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2, "kpc"))
    >>> chart.components
    ('mu', 'nu', 'phi')

    >>> chart.coord_dimensions
    ('area', 'area', 'angle')

    >>> isinstance(cxc.cartesian_chart(chart), cxc.Cart3D)
    True

    >>> d = {"mu": u.Q(4, "kpc2"), "nu": u.Q(0.1, "kpc2"), "phi": u.Q(1, "rad")}
    >>> chart.check_data(d)  # doesn't raise
    {'mu': Q(4, 'kpc2'), 'nu': Q(0.1, 'kpc2'), 'phi': Q(1, 'rad')}

    """

    _: dataclasses.KW_ONLY
    Delta: Annotated[
        Real[u.quantity.StaticQuantity, ""],  # StaticQuantity["length"]
        Is[lambda x: x.value > 0],
    ]
    """Focal length of the coordinate system."""

    M: MT = R3  # ty: ignore[invalid-assignment]

    def check_data(self, data: CDictT, /, *, values: bool = False, **kw: Any) -> CDictT:
        super().check_data(data, **kw)  # call base check
        checks.strictly_positive(self.Delta, name="Delta")
        if values:
            checks.geq(data["mu"], self.Delta**2, name="mu", comp_name="Delta^2")
            checks.leq(
                jnp.abs(data["nu"]), self.Delta**2, name="nu", comp_name="Delta^2"
            )
        return data

    @override
    @property
    def cartesian(self) -> Cart3D[MT]:
        """Return the canonical Cartesian chart for a 3D chart.

        >>> import coordinax.charts as cxc
        >>> import unxt as u

        >>> chart = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2, "kpc"))
        >>> isinstance(chart.cartesian, cxc.Cart3D)
        True

        """
        return Cart3D(M=self.M)
