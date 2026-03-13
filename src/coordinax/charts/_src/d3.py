"""Vector."""

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

from dataclasses import KW_ONLY

from jaxtyping import Real
from typing import Annotated, Any, Final, Literal as L, final  # noqa: N817
from typing_extensions import override

import plum
from beartype.vale import Is

import quaxed.numpy as jnp
import unxt as u

from . import checks
from .base import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
)
from .constants import Deg0, Deg90, Deg180
from .custom_types import CDict
from .utils import is_not_abstract_chart_subclass
from coordinax.internal.custom_types import Ang, Len


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


@final
@chart_dataclass_decorator
class Cart3D(AbstractFixedComponentsChart[Cart3DKeys, Cart3DDims], Abstract3D):
    pass


cart3d: Final = Cart3D()


@plum.dispatch
def cartesian_chart(obj: Cart3D, /) -> Cart3D:
    return cart3d


# -----------------------------------------------
# Cylindrical

CylindricalKeys = tuple[L["rho"], L["phi"], L["z"]]
Cylindrical3DDims = tuple[Len, Ang, Len]


@final
@chart_dataclass_decorator
class Cylindrical3D(
    AbstractFixedComponentsChart[CylindricalKeys, Cylindrical3DDims], Abstract3D
):
    pass


cyl3d: Final = Cylindrical3D()


@plum.dispatch
def cartesian_chart(obj: Cylindrical3D, /) -> Cart3D:
    return cart3d


# -----------------------------------------------
# Spherical


class AbstractSpherical3D(Abstract3D):
    """Abstract spherical vector representation."""


@plum.dispatch
def cartesian_chart(obj: AbstractSpherical3D, /) -> Cart3D:
    return cart3d


SphericalKeys = tuple[L["r"], L["theta"], L["phi"]]
Spherical3DDims = tuple[Len, Ang, Ang]


@final
@chart_dataclass_decorator
class Spherical3D(
    AbstractFixedComponentsChart[SphericalKeys, Spherical3DDims], AbstractSpherical3D
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

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["theta"])


sph3d: Final = Spherical3D()


# -----------------------------------------------
# LonLatSpherical

LonLatSphericalKeys = tuple[L["lon"], L["lat"], L["distance"]]
LonLatSpherical3DDims = tuple[Ang, Ang, Len]


@final
@chart_dataclass_decorator
class LonLatSpherical3D(
    AbstractFixedComponentsChart[LonLatSphericalKeys, LonLatSpherical3DDims],
    AbstractSpherical3D,
):
    r"""Longitude-latitude spherical coordinates.

    Components are $(\mathrm{lon}, \mathrm{lat}, r)$ where:

    - ``lon`` is the azimuthal angle in the equatorial plane,
    - ``lat`` is the latitude in $[-\pi/2, \pi/2]$,
    - ``r`` is the radial distance.

    Relation to standard spherical coordinates:

    $\mathrm{lat} = \pi/2 - \theta$,  $\mathrm{lon} = \\phi$.
    """

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["lat"], -Deg90, Deg90)


lonlat_sph3d: Final = LonLatSpherical3D()

# -----------------------------------------------
# LonCosLatSpherical

LonCosLatSphericalKeys = tuple[L["lon_coslat"], L["lat"], L["distance"]]
LonCosLatSpherical3DDims = tuple[Ang, Ang, Len]


@final
@chart_dataclass_decorator
class LonCosLatSpherical3D(
    AbstractFixedComponentsChart[LonCosLatSphericalKeys, LonCosLatSpherical3DDims],
    AbstractSpherical3D,
):
    r"""Longitude-cos(latitude) spherical coordinates.

    Components are $(\mathrm{lon}\cos\mathrm{lat}, \mathrm{lat}, r)$.
    This form is sometimes used to improve numerical behavior near the poles,
    since $\cos(\mathrm{lat}) \to 0$ as $|\mathrm{lat}| \to \pi/2$.
    """

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["lat"], -Deg90, Deg90)


loncoslat_sph3d: Final = LonCosLatSpherical3D()

# -----------------------------------------------
# MathSpherical

MathSphericalKeys = tuple[L["r"], L["theta"], L["phi"]]


@final
@chart_dataclass_decorator
class MathSpherical3D(
    AbstractFixedComponentsChart[MathSphericalKeys, Spherical3DDims],
    AbstractSpherical3D,
):
    r"""Mathematical-convention spherical coordinates $(r, \theta, \phi)$.

    In this convention:

    - $\phi$ is the polar angle measured from the positive $z$ axis,
    - $\theta$ is the azimuthal angle in the $xy$-plane.

    This differs from the physics convention used by
    :class:`Spherical3D`, where $\theta$ and $\phi$ are swapped.
    """

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["phi"], Deg0, Deg180)


math_sph3d: Final = MathSpherical3D()

# -----------------------------------------------
# Prolate Spheroidal

ProlateSpheroidalKeys = tuple[L["mu"], L["nu"], L["phi"]]
ProlateSpheroidal3DDims = tuple[L["area"], L["area"], Ang]


@final
@chart_dataclass_decorator
class ProlateSpheroidal3D(
    AbstractFixedComponentsChart[ProlateSpheroidalKeys, ProlateSpheroidal3DDims],
    AbstractSpherical3D,
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
    Delta: Annotated[
        Real[u.quantity.StaticQuantity, ""],  # StaticQuantity["length"]
        Is[lambda x: x.value > 0],
    ]
    """Focal length of the coordinate system."""

    def check_data(self, data: CDict, /) -> None:
        super().check_data(data)  # call base check
        checks.strictly_positive(self.Delta, name="Delta")
        checks.geq(data["mu"], self.Delta**2, name="mu", comp_name="Delta^2")
        checks.leq(jnp.abs(data["nu"]), self.Delta**2, name="nu", comp_name="Delta^2")


@plum.dispatch
def cartesian_chart(obj: ProlateSpheroidal3D, /) -> Cart3D:
    return cart3d
