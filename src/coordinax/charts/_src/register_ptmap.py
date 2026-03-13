"""Point-roled transformations in the same atlas."""

__all__: tuple[str, ...] = ()


from collections.abc import Callable, Mapping
from jaxtyping import Array
from typing import Any, Final, final

import equinox as eqx
import jax
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as ABCQ  # noqa: N814

import coordinax.api.charts as api
from .base import AbstractChart
from .custom_types import CDict, CKey
from .d0 import Abstract0D, Cart0D
from .d1 import Abstract1D, Cart1D, Radial1D, Time1D
from .d2 import Cart2D, Polar2D
from .d3 import (
    AbstractSpherical3D,
    Cart3D,
    Cylindrical3D,
    LonCosLatSpherical3D,
    LonLatSpherical3D,
    MathSpherical3D,
    ProlateSpheroidal3D,
    Spherical3D,
    cyl3d,
    sph3d,
)
from .d6 import PoincarePolar6D
from .dn import CartND
from .sphere2 import (
    LonCosLatSphericalTwoSphere,
    LonLatSphericalTwoSphere,
    MathSphericalTwoSphere,
    SphericalTwoSphere,
)
from .utils import uconvert_to_rad
from coordinax.internal import QuantityMatrix, UnitsMatrix
from coordinax.internal.custom_types import OptUSys


@final
class MissingType:
    """Sentinel for missing arguments."""


MISSING: Final[MissingType] = MissingType()


#####################################################################
# Point transformations

# ===================================================================
# Partial application


@plum.dispatch
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    /,
    **fixed_kwargs: Any,
) -> Callable[..., Any]:
    """Return a partial function for point transformation.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> transform_func = cxc.point_transition_map(cxc.sph3d, cxc.cart3d)
    >>> transform_func(p)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    >>> p = {"x": 1.0, "y": 0.0, "z": 0.0}  # No units
    >>> transform_func = cxc.point_transition_map(cxc.sph3d, cxc.cart3d,
    ...                                      usys=u.unitsystems.si)
    >>> transform_func(p)
    {'r': Array(1., dtype=float64, weak_type=True),
     'theta': Array(1.57079633, dtype=float64),
     'phi': Array(0., dtype=float64, weak_type=True)}

    """
    # NOTE: lambda is much faster than ft.partial here
    return lambda *args, **kw: api.point_transition_map(
        to_chart, from_chart, *args, **fixed_kwargs, **kw
    )


##############################################################################
# Transition Maps Assuming on Same Atlas

# ===================================================================
# Self representation conversions

IDENTITY_TRANSFORM_CHARTS: Final[tuple[type[AbstractChart[Any, Any]], ...]] = (
    # 0D
    Cart0D,
    # 1D
    Cart1D,
    Radial1D,
    Time1D,
    # 2D
    Cart2D,
    Polar2D,
    SphericalTwoSphere,
    LonLatSphericalTwoSphere,
    LonCosLatSphericalTwoSphere,
    MathSphericalTwoSphere,
    # 3D
    Cart3D,
    Cylindrical3D,
    Spherical3D,
    LonLatSpherical3D,
    LonCosLatSpherical3D,
    MathSpherical3D,
    # ProlateSpheroidal3D,  # requires Delta
    # 6D
    PoincarePolar6D,
    # N-D
    CartND,
    # SpaceTimeCT,  # depends on spatial chart
)


@plum.dispatch.multi(
    *((typ, typ, dict[CKey, Any]) for typ in IDENTITY_TRANSFORM_CHARTS)
)
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Identity conversion for matching charts.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> p2 = cxc.point_transition_map(cxc.cart2d, cxc.cart2d, p)
    >>> p is p2
    True

    >>> p = {"r": u.Q(3.0, "m")}
    >>> p2 = cxc.point_transition_map(cxc.radial1d, cxc.radial1d, p)
    >>> p is p2
    True

    """
    del to_chart, from_chart, usys  # unused
    return p


# ---------------------------------------------------------
# General representation conversions


@plum.dispatch(precedence=-1)
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> Cartesian -> AbstractChart."""
    p_cart = api.point_transition_map(to_chart.cartesian, from_chart, p, usys=usys)
    p_out = api.point_transition_map(to_chart, from_chart.cartesian, p_cart, usys=usys)
    return p_out  # noqa: RET504


# ---------------------------------------------------------
# Specific representation conversions

# -----------------------------------------------
# 1D


@plum.dispatch
def point_transition_map(
    to_chart: Cart1D, from_chart: Radial1D, p: CDict, /, usys: OptUSys = None
) -> CDict:
    """Radial1D -> Cart1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m")}
    >>> cxc.point_transition_map(cxc.cart1d, cxc.radial1d, p)
    {'x': Q(5., 'm')}

    >>> p = {"r": 5.0}  # No units
    >>> cxc.point_transition_map(cxc.cart1d, cxc.radial1d, p)
    {'x': 5.0}

    """
    del to_chart, from_chart, usys  # unused
    return {"x": p["r"]}


@plum.dispatch
def point_transition_map(
    to_chart: Radial1D, from_chart: Cart1D, p: CDict, /, usys: OptUSys = None
) -> CDict:
    """Cart1D -> Radial1D.

    The `x` coordinate is converted to the `r` coordinate of the 1D system.

    Assumptions:

    - Cart1D and Radial1D are

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(5.0, "m")}
    >>> cxc.point_transition_map(cxc.radial1d, cxc.cart1d, p)
    {'r': Q(5., 'm')}

    >>> p = {"x": 5.0}  # No units
    >>> cxc.point_transition_map(cxc.radial1d, cxc.cart1d, p)
    {'r': 5.0}

    """
    del to_chart, from_chart, usys  # unused
    return {"r": p["x"]}


# -----------------------------------------------
# 2D


@plum.dispatch
def point_transition_map(
    to_chart: Cart2D, from_chart: Polar2D, p: CDict, /, usys: OptUSys = None
) -> CDict:
    """Polar2D -> Cart2D.

    The `r` and `theta` coordinates are converted to the `x` and `y` coordinates
    of the 2D Cartesian system.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxc.point_transition_map(cxc.cart2d, cxc.polar2d, p)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    >>> p = {"r": 5, "theta": 90}  # No units
    >>> usys = u.unitsystem("km", "deg")
    >>> cxc.point_transition_map(cxc.cart2d, cxc.polar2d, p, usys=usys)
    {'x': Array(3.061617e-16, dtype=float64, ...),
     'y': Array(5., dtype=float64, ...)}

    """
    del to_chart, from_chart  # unused
    theta = uconvert_to_rad(p["theta"], usys)
    x = p["r"] * jnp.cos(theta)
    y = p["r"] * jnp.sin(theta)
    return {"x": x, "y": y}


@plum.dispatch
def point_transition_map(
    to_chart: Polar2D, from_chart: Cart2D, p: CDict, /, usys: OptUSys = None
) -> CDict:
    """Cart2D -> Polar2D.

    The `x` and `y` coordinates are converted to the `r` and `theta` coordinates
    of the 2D polar system.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(3, "m"), "y": u.Q(4, "m")}
    >>> cxc.point_transition_map(cxc.polar2d, cxc.cart2d, p)
    {'r': Q(5., 'm'), 'theta': Q(0.92729522, 'rad')}

    >>> p = {"x": 3, "y": 4}  # No units
    >>> cxc.point_transition_map(cxc.polar2d, cxc.cart2d, p)
    {'r': Array(5., dtype=float64, ...),
     'theta': Array(0.92729522, dtype=float64, ...)}

    """
    del to_chart, from_chart, usys  # unused
    r_ = jnp.hypot(p["x"], p["y"])
    theta = jnp.arctan2(p["y"], p["x"])
    return {"r": r_, "theta": theta}


# -----------------------------------------------
# 3D


@plum.dispatch
def point_transition_map(
    to_chart: Cart3D,
    from_chart: Cylindrical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Cylindrical3D -> Cart3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"rho": u.Q(1.0, "m"), "phi": u.Q(90, "deg"), "z": u.Q(2.0, "m")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.cyl3d, p)
    {'x': Q(6.123234e-17, 'm'), 'y': Q(1., 'm'), 'z': Q(2., 'm')}

    >>> p = {"rho": 1.0, "phi": 90, "z": 2.0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.point_transition_map(cxc.cart3d, cxc.cyl3d, p, usys=usys)
    {'x': Array(6.123234e-17, dtype=float64, ...), 'y': Array(1., dtype=float64, ...),
     'z': 2.0}

    """
    del to_chart, from_chart  # unused
    phi = uconvert_to_rad(p["phi"], usys)
    x = p["rho"] * jnp.cos(phi)
    y = p["rho"] * jnp.sin(phi)
    return {"x": x, "y": y, "z": p["z"]}


@plum.dispatch
def point_transition_map(
    to_chart: Cart3D,
    from_chart: Spherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> Cart3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.sph3d, p)
    {'x': Q(0., 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the equator (theta=90 deg, phi=0):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.point_transition_map(cxc.cart3d, cxc.sph3d, p, usys=usys)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    del to_chart, from_chart  # unused
    r_ = p["r"]
    theta = uconvert_to_rad(p["theta"], usys)
    phi = uconvert_to_rad(p["phi"], usys)
    x = r_ * jnp.sin(theta) * jnp.cos(phi)
    y = r_ * jnp.sin(theta) * jnp.sin(phi)
    z = r_ * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transition_map(
    to_chart: Cart3D,
    from_chart: LonLatSpherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """LonLatSpherical3D -> Cart3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point at the north pole (lat=90 deg):

    >>> p = {"lon": u.Q(0, "deg"), "lat": u.Q(90, "deg"), "distance": u.Q(1.0, "m")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.lonlat_sph3d, p)
    {'x': Q(6.123234e-17, 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the equator at lon=0:

    >>> p = {"lon": 0, "lat": 0, "distance": 2}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.lonlat_sph3d, p)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(0., dtype=float64, ...)}

    """
    del to_chart, from_chart  # unused
    r_ = p["distance"]
    lon = uconvert_to_rad(p["lon"], usys)
    lat = uconvert_to_rad(p["lat"], usys)
    x = r_ * jnp.cos(lat) * jnp.cos(lon)
    y = r_ * jnp.cos(lat) * jnp.sin(lon)
    z = r_ * jnp.sin(lat)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transition_map(
    to_chart: Cart3D,
    from_chart: LonCosLatSpherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """LonCosLatSpherical3D -> Cart3D.

    Components are (lon_coslat, lat, r), where lon_coslat := lon * cos(lat).
    Longitude is undefined at the poles (cos(lat) == 0); we set lon = 0 by
    convention there to avoid NaNs.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the equator (lat=0, so lon_coslat = lon):

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(0, "deg"),
    ...      "distance": u.Q(1.0, "m")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.loncoslat_sph3d, p)
    {'x': Q(1., 'm'), 'y': Q(0., 'm'), 'z': Q(0., 'm')}

    At the north pole (lat=90), lon_coslat is effectively 0 regardless of lon:

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(90, "deg"),
    ...      "distance": u.Q(2.0, "m")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.loncoslat_sph3d, p)
    {'x': Q(1.2246468e-16, 'm'), 'y': Q(0., 'm'), 'z': Q(2., 'm')}

    """
    del to_chart, from_chart  # unused
    lon_coslat, r_ = p["lon_coslat"], p["distance"]
    lat = uconvert_to_rad(p["lat"], usys)
    # Handle the poles where cos(lat) == 0
    coslat = jnp.cos(lat)
    lon = jnp.where(coslat == 0, 0, lon_coslat / coslat)
    lon = uconvert_to_rad(lon, usys)
    # Convert to Cartesian
    x = r_ * jnp.cos(lat) * jnp.cos(lon)
    y = r_ * jnp.cos(lat) * jnp.sin(lon)
    z = r_ * jnp.sin(lat)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transition_map(
    to_chart: Cart3D,
    from_chart: MathSpherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """MathSpherical3D -> Cart3D.

    - theta: azimuth in the x-y plane (longitude-like)
    - phi  : polar angle from +z, with phi in [0, pi]

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the +z axis (phi=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.math_sph3d, p)
    {'x': Q(0., 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the +x axis (theta=0, phi=90):

    >>> p = {"r": u.Q(2.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(90, "deg")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.math_sph3d, p)
    {'x': Q(2., 'm'), 'y': Q(0., 'm'), 'z': Q(1.2246468e-16, 'm')}

    """
    del to_chart, from_chart  # unused
    r_ = p["r"]
    theta = uconvert_to_rad(p["theta"], usys)
    phi = uconvert_to_rad(p["phi"], usys)
    x = r_ * jnp.sin(phi) * jnp.cos(theta)
    y = r_ * jnp.sin(phi) * jnp.sin(theta)
    z = r_ * jnp.cos(phi)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transition_map(
    to_chart: Cart3D,
    from_chart: ProlateSpheroidal3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    r"""ProlateSpheroidal3D -> Cart3D.

    We calculate through cylindrical coordinates first:

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{|\nu|}{\Delta^2}\right)}$
    $z = \sqrt{\mu\,\frac{|\nu|}{\Delta^2}}\;\mathrm{sign}(\nu)$
    $\phi = \phi.$

    Then convert to Cartesian:

    $x=\rho\cos\phi$, $y=\rho\sin\phi$, $z=z$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.point_transition_map(cxc.cart3d, prolatesph3d, p)
    {'x': Q(0.8660254, 'm'), 'y': Q(0., 'm'), 'z': Q(1.11803399, 'm')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.point_transition_map(cxc.cart3d, prolatesph3d, p, usys=usys)
    {'x': Array(0.8660254, dtype=float64),
     'y': Array(0., dtype=float64),
     'z': Array(1.11803399, dtype=float64)}

    """
    del to_chart
    # Calculate cylindrical distance
    nu, mu = p["nu"], p["mu"]
    if not isinstance(nu, ABCQ) or not isinstance(mu, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'mu' or 'nu', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = u.ustrip(usys["length"], from_chart.Delta) ** 2
    else:
        # TODO: fix Delta**2 issue in unxt
        Delta2 = u.Q.from_(from_chart.Delta) ** 2

    nu_D2 = jnp.abs(nu) / Delta2
    rho = jnp.sqrt((mu - Delta2) * (1 - nu_D2))
    # Convert to Cartesian
    phi = uconvert_to_rad(p["phi"], usys)
    x = rho * jnp.cos(phi)
    y = rho * jnp.sin(phi)
    z = jnp.sqrt(mu * nu_D2) * jnp.sign(nu)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def point_transition_map(
    to_chart: Cylindrical3D,
    from_chart: Cart3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Cylindrical3D.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import unxt as u

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxc.point_transition_map(cxc.cyl3d, cxc.cart3d, p)
    {'rho': Q(5., 'm'), 'phi': Q(0.92729522, 'rad'), 'z': Q(5., 'm')}

    >>> p = {"x": 3.0, "y": 4.0, "z": 5.0}  # No units
    >>> cxc.point_transition_map(cxc.cyl3d, cxc.cart3d, p)
    {'rho': Array(5., dtype=float64, ...),
     'phi': Array(0.92729522, dtype=float64, ...),
     'z': 5.0}

    """
    del to_chart, from_chart, usys  # Unused
    rho = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    return {"rho": rho, "phi": phi, "z": p["z"]}


@plum.dispatch.multi(
    (AbstractSpherical3D, Cart3D, Mapping),
    (AbstractSpherical3D, Cylindrical3D, Mapping),
)
def point_transition_map(
    to_chart: AbstractSpherical3D,
    from_chart: Cart3D | Cylindrical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Spherical3D -> AbstractSpherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxc.point_transition_map(cxc.loncoslat_sph3d, cxc.cart3d, p)
    {'lon_coslat': Q(0., 'rad'), 'lat': Q(90., 'deg'), 'distance': Q(1., 'm')}

    >>> p = {"rho": 0, "phi": 180, "z": 1}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.point_transition_map(cxc.loncoslat_sph3d, cxc.cyl3d, p, usys=usys)
    {'lon_coslat': Array(1.10218212e-14, dtype=float64),
     'lat': Array(1.57079633, dtype=float64),
     'distance': Array(1., dtype=float64, weak_type=True)}

    """
    p_sph = api.point_transition_map(sph3d, from_chart, p, usys=usys)
    return api.point_transition_map(to_chart, sph3d, p_sph, usys=usys)


@plum.dispatch
def point_transition_map(
    to_chart: Spherical3D,
    from_chart: Cart3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Spherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the +z axis:

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxc.point_transition_map(cxc.sph3d, cxc.cart3d, p)
    {'r': Q(1., 'm'), 'theta': Q(0., 'rad'), 'phi': Q(0., 'rad')}

    A point on the +x axis:

    >>> p = {"x": 2.0, "y": 0.0, "z": 0.0}  # No units
    >>> cxc.point_transition_map(cxc.sph3d, cxc.cart3d, p)
    {'r': Array(2., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64),
     'phi': Array(0., dtype=float64, ...)}

    """
    del to_chart, from_chart  # unused
    x, y, z = p["x"], p["y"], p["z"]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r == 0, jnp.ones(r.shape), z / r))
    # atan2 handles the case when x = y = 0, returning phi = 0
    phi = jnp.atan2(y, x)
    return {"r": r, "theta": theta, "phi": phi}


@plum.dispatch
def point_transition_map(
    to_chart: Spherical3D,
    from_chart: Cylindrical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Cylindrical3D -> Spherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(1.0, "m")}
    >>> cxc.point_transition_map(cxc.sph3d, cxc.cyl3d, p)
    {'r': Q(1., 'm'), 'theta': Q(0., 'rad'), 'phi': Q(0, 'rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": 3.0, "phi": 0, "z": 0.0}  # No units
    >>> cxc.point_transition_map(cxc.sph3d, cxc.cyl3d, p)
    {'r': Array(3., dtype=float64, ...), 'theta': Array(1.57079633, dtype=float64),
     'phi': 0}

    """
    del to_chart, from_chart  # unused
    r_ = jnp.hypot(p["rho"], p["z"])
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r_ == 0, jnp.ones(r_.shape), p["z"] / r_))
    return {"r": r_, "theta": theta, "phi": p["phi"]}


@plum.dispatch
def point_transition_map(
    to_chart: Cylindrical3D,
    from_chart: Spherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> Cylindrical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxc.point_transition_map(cxc.cyl3d, cxc.sph3d, p)
    {'rho': Q(0., 'm'), 'phi': Q(0, 'rad'), 'z': Q(1., 'm')}

    A point on the equator (theta=90 deg):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.point_transition_map(cxc.cyl3d, cxc.sph3d, p, usys=usys)
    {'rho': Array(2., dtype=float64, ...), 'phi': 0,
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    del to_chart, from_chart  # unused
    theta = uconvert_to_rad(p["theta"], usys)
    rho = p["r"] * jnp.sin(theta)
    z = p["r"] * jnp.cos(theta)
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def point_transition_map(
    to_chart: LonLatSpherical3D,
    from_chart: Spherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> LonLatSpherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Spherical theta=0 corresponds to lat=90 (north pole):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxc.point_transition_map(cxc.lonlat_sph3d, cxc.sph3d, p)
    {'lon': Q(0, 'rad'), 'lat': Q(90., 'deg'), 'distance': Q(1., 'm')}

    Spherical theta=90 deg corresponds to lat=0 (equator):

    >>> p = {"r": 1.0, "theta": 0, "phi": 0}  # No units
    >>> cxc.point_transition_map(cxc.lonlat_sph3d, cxc.sph3d, p)
    {'lon': 0, 'lat': 1.5707963267948966, 'distance': 1.0}

    """
    del to_chart, from_chart  # unused
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    return {"lon": p["phi"], "lat": lat, "distance": p["r"]}


@plum.dispatch
def point_transition_map(
    to_chart: LonCosLatSpherical3D,
    from_chart: Spherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> LonCosLatSpherical3D.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    On the equator (theta=90 deg), lon_coslat equals lon:

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}
    >>> cxc.point_transition_map(cxc.loncoslat_sph3d, cxc.sph3d, p)
    {'lon_coslat': Q(45., 'deg'), 'lat': Q(0., 'deg'), 'distance': Q(1., 'm')}

    At the north pole (theta=0), lon_coslat = 0 regardless of phi:

    >>> p = {"r": 1.0, "theta": 0, "phi": 45}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.point_transition_map(cxc.loncoslat_sph3d, cxc.sph3d, p, usys=usys)
    {'lon_coslat': Array(2.7554553e-15, dtype=float64, ...),
     'lat': 1.5707963267948966, 'distance': 1.0}

    """
    del to_chart, from_chart  # unused
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat)
    return {"lon_coslat": lon_coslat, "lat": lat, "distance": p["r"]}


@plum.dispatch
def point_transition_map(
    to_chart: MathSpherical3D,
    from_chart: Spherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> MathSpherical3D.

    Swaps theta and phi: Physics (theta=polar, phi=azimuth) to
    Math (theta=azimuth, phi=polar).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
    >>> cxc.point_transition_map(cxc.math_sph3d, cxc.sph3d, p)
    {'r': Q(1., 'm'), 'theta': Q(60, 'deg'), 'phi': Q(30, 'deg')}

    >>> p = {"r": 1.0, "theta": 30, "phi": 60}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.point_transition_map(cxc.math_sph3d, cxc.sph3d, p, usys=usys)
    {'r': 1.0, 'theta': 60, 'phi': 30}

    """
    del to_chart, from_chart, usys  # Unused
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def point_transition_map(
    to_chart: Spherical3D,
    from_chart: MathSpherical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """MathSpherical3D -> Spherical3D.

    Swaps theta and phi: Math (theta=azimuth, phi=polar) to
    Physics (theta=polar, phi=azimuth).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(60, "deg"), "phi": u.Q(30, "deg")}
    >>> cxc.point_transition_map(cxc.sph3d, cxc.math_sph3d, p)
    {'r': Q(1., 'm'), 'theta': Q(30, 'deg'), 'phi': Q(60, 'deg')}

    >>> p = {"r": 1.0, "theta": 60, "phi": 30}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.point_transition_map(cxc.sph3d, cxc.math_sph3d, p, usys=usys)
    {'r': 1.0, 'theta': 30, 'phi': 60}

    """
    del to_chart, from_chart, usys  # Unused
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


def point_transition_map(
    to_chart: Cylindrical3D,
    from_chart: ProlateSpheroidal3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    r"""ProlateSpheroidal3D -> Cylindrical3D.

    Uses the focal length $\Delta$ stored on ``from_chart``.

    Validity constraints (enforced by the representation) are:

    - $\Delta > 0$,
    - $\mu \ge \Delta^2$,
    - $|\nu| \le \Delta^2$.

    The conversion proceeds via

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{|\nu|}{\Delta^2}\right)}$,
    $z = \sqrt{\mu\,\frac{|\nu|}{\Delta^2}}\,\mathrm{sign}(\nu)$,
    $\phi = \phi$.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.point_transition_map(cxc.cyl3d, prolatesph3d, p)
    {'rho': Q(0.8660254, 'm'), 'phi': Q(0., 'rad'), 'z': Q(1.11803399, 'm')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.point_transition_map(cxc.cyl3d, prolatesph3d, p, usys=usys)
    {'rho': Array(0.8660254, dtype=float64), 'phi': Array(0., dtype=float64),
     'z': Array(1.11803399, dtype=float64)}

    """
    del to_chart  # Unused
    nu, mu = p["nu"], p["mu"]
    if not isinstance(nu, ABCQ) or not isinstance(mu, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'mu' or 'nu', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = u.ustrip(usys["area"], from_chart.Delta**2)
    else:
        Delta2 = from_chart.Delta**2
    nu_D2 = jnp.abs(nu) / Delta2
    rho = jnp.sqrt((mu - Delta2) * (1 - nu_D2))
    z = jnp.sqrt(mu * nu_D2) * jnp.sign(nu)
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def point_transition_map(
    to_chart: ProlateSpheroidal3D,
    from_chart: Cylindrical3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    r"""Cylindrical3D -> ProlateSpheroidal3D.

    Uses the focal length $\Delta$ stored on ``to_chart``.

    Let $R^2 = \rho^2$ and $z^2 = z^2$ and define

    $S = R^2 + z^2 + \Delta^2$,
    $D_f = R^2 + z^2 - \Delta^2$,
    $D = \sqrt{D_f^2 + 4 R^2 \Delta^2}$.

    Then

    $\mu = \Delta^2 + \tfrac12(D + D_f)$ (with numerically-stable branches),
    $|\nu| = \dfrac{2\Delta^2}{S + D}\,z^2$,
    and $\nu = |\nu|\,\mathrm{sign}(z)$, with a stability fix when
    $\Delta^2 - |\nu|$ is small.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(3.0, "m")}
    >>> cxc.point_transition_map(prolatesph3d, cxc.cyl3d, p)
    {'mu': Q(9., 'm2'), 'nu': Q(4., 'm2'), 'phi': Q(0, 'rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": u.Q(2.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(0.0, "m")}
    >>> cxc.point_transition_map(prolatesph3d, cxc.cyl3d, p)
    {'mu': Q(8., 'm2'), 'nu': Q(0., 'm2'), 'phi': Q(0, 'rad')}

    Without units:

    >>> p = {"rho": 2.0, "phi": 0, "z": 3.0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.point_transition_map(prolatesph3d, cxc.cyl3d, p, usys=usys)
    {'mu': Array(14.52079729, dtype=float64),
     'nu': Array(2.47920271, dtype=float64), 'phi': 0}

    """
    del from_chart  # Unused
    # Pre-compute common terms
    R2 = p["rho"] ** 2
    z2 = p["z"] ** 2
    if not isinstance(R2, ABCQ) or not isinstance(z2, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'rho' or 'z', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = u.ustrip(usys["length"], to_chart.Delta) ** 2
    else:
        # TODO: fix Delta**2 issue in unxt
        Delta2 = u.Q.from_(to_chart.Delta) ** 2

    sum_ = R2 + z2 + Delta2
    diff_ = R2 + z2 - Delta2

    # D = sqrt((R^2 + z^2 - Delta^2)^2 + 4 R^2 Delta^2)
    D = jnp.sqrt(diff_**2 + 4 * R2 * Delta2)

    # Handle special cases for z=0 or rho=0
    D = jnp.where(p["z"] == 0, sum_, D)
    D = jnp.where(p["rho"] == 0, jnp.abs(diff_), D)

    # Numerically stable branches (avoid dividing by small numbers)
    pos_mu_minus_delta = 0.5 * (D + diff_)
    pos_delta_minus_nu = Delta2 * R2 / pos_mu_minus_delta

    neg_delta_minus_nu = 0.5 * (D - diff_)
    neg_mu_minus_delta = Delta2 * R2 / neg_delta_minus_nu

    mu_minus_delta = jnp.where(diff_ >= 0, pos_mu_minus_delta, neg_mu_minus_delta)
    delta_minus_nu = jnp.where(diff_ >= 0, pos_delta_minus_nu, neg_delta_minus_nu)

    mu = Delta2 + mu_minus_delta

    # |nu| = 2 Delta^2 / (sum_ + D) * z^2
    abs_nu = 2 * Delta2 / (sum_ + D) * z2

    # Stability fix when Delta^2 - |nu| is small
    abs_nu = jnp.where(abs_nu * 2 > Delta2, Delta2 - delta_minus_nu, abs_nu)

    nu = abs_nu * jnp.sign(p["z"])

    return {"mu": mu, "nu": nu, "phi": p["phi"]}


@plum.dispatch
def point_transition_map(
    to_chart: ProlateSpheroidal3D,
    from_chart: ProlateSpheroidal3D,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    r"""{class}`coordinax.charts.ProlateSpheroidal3D` -> itself.

    If the focal length is unchanged (``to_chart.Delta == from_chart.Delta``), this
    is the identity map.

    If the focal length changes, we convert via cylindrical coordinates:

    ``Prolate(Delta_in) -> Cylindrical -> Prolate(Delta_out)``.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Same focal length (identity transformation):

    >>> prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.point_transition_map(prolate, prolate, p)
    {'mu': Q(5., 'm2'),
     'nu': Q(1., 'm2'),
     'phi': Q(0., 'rad')}

    Different focal lengths (converts via cylindrical):

    >>> prolate_in = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> prolate_out = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(3.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.point_transition_map(prolate_out, prolate_in, p)
    {'mu': Q(9.85889894, 'm2'),
     'nu': Q(1.14110106, 'm2'),
     'phi': Q(0., 'rad')}

    """
    # Cast to the result type
    dtype = jnp.result_type(
        to_chart.Delta, from_chart.Delta, *[v.dtype for v in p.values()]
    )
    p = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), p)
    return jax.lax.cond(
        to_chart.Delta == from_chart.Delta,
        lambda p: p,
        lambda p: api.point_transition_map(
            to_chart,
            cyl3d,
            api.point_transition_map(cyl3d, from_chart, p, usys=usys),
            usys=usys,
        ),
        p,
    )


# -----------------------------------------------
# N-D


def _cartnd_to_cartesian(
    p: CDict,
    cart_chart: AbstractChart,  # type: ignore[type-arg]
) -> CDict:
    """Convert CartND data dict to a fixed-dimensional Cartesian dict.

    Extracts components from the 'q' array and maps them to named components.
    """
    q = p["q"]
    return {comp: q[i] for i, comp in enumerate(cart_chart.components)}


@plum.dispatch
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: CartND,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """CartND -> AbstractChart.

    Converts from N-dimensional Cartesian (with a single 'q' array) to any
    other chart type by first extracting the appropriate fixed-dimensional
    Cartesian representation.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Convert 3D CartND to Spherical:

    >>> p = {"q": u.Q([1.0, 0.0, 0.0], "m")}
    >>> cxc.point_transition_map(cxc.sph3d, cxc.cartnd, p)
    {'r': Q(1., 'm'),
     'theta': Q(1.57079633, 'rad'),
     'phi': Q(0., 'rad')}

    Convert 2D CartND to Polar:

    >>> p = {"q": u.Q([3.0, 4.0], "m")}
    >>> cxc.point_transition_map(cxc.polar2d, cxc.cartnd, p)
    {'r': Q(5., 'm'),
     'theta': Q(0.92729522, 'rad')}

    Convert 1D CartND to Radial:

    >>> p = {"q": u.Q([5.0], "m")}
    >>> cxc.point_transition_map(cxc.radial1d, cxc.cartnd, p)
    {'r': Q(5., 'm')}

    Convert CartND to Cart3D:

    >>> p = {"q": u.Q([1.0, 2.0, 3.0], "m")}
    >>> cxc.point_transition_map(cxc.cart3d, cxc.cartnd, p)
    {'x': Q(1., 'm'),
     'y': Q(2., 'm'),
     'z': Q(3., 'm')}

    """
    del from_chart  # Unused

    # If target is CartND, we can't convert (would be infinite recursion)
    if isinstance(to_chart, CartND):
        msg = "Cannot convert CartND to CartND via this dispatch."
        raise TypeError(msg)

    # Get the corresponding fixed-dimensional Cartesian chart
    cart_chart = to_chart.cartesian

    # If cartesian_chart returns CartND, we don't support this conversion
    if isinstance(cart_chart, CartND):
        msg = f"CartND conversion not supported for {type(to_chart).__name__}."
        raise NotImplementedError(msg)

    # Get dimensionality from the target chart
    target_ndim = to_chart.ndim

    # Check that the CartND data has the right dimensionality
    q = p["q"]
    data_ndim = q.shape[0]
    eqx.error_if(
        q,
        data_ndim != target_ndim,
        f"CartND data has {data_ndim} dimensions but target chart "
        f"{type(to_chart).__name__} requires {target_ndim} dimensions.",
    )

    # Convert CartND to fixed-dimensional Cartesian
    p_cart = _cartnd_to_cartesian(p, cart_chart)

    # If target is already the Cartesian chart, return directly
    if type(to_chart) is type(cart_chart):
        return p_cart

    # Otherwise, transform from Cartesian to target chart
    return api.point_transition_map(to_chart, cart_chart, p_cart, usys=usys)


def _cartesian_to_cartnd(
    p: CDict,
    cart_chart: AbstractChart,  # type: ignore[type-arg]
) -> CDict:
    """Convert a fixed-dimensional Cartesian dict to CartND data dict.

    Stacks named components into a single 'q' array.
    """
    q = jnp.stack([p[comp] for comp in cart_chart.components], axis=0)
    return {"q": q}


@plum.dispatch
def point_transition_map(
    to_chart: CartND,
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> CartND.

    Converts from any chart type to N-dimensional Cartesian (with a single
    'q' array) by first transforming to the appropriate fixed-dimensional
    Cartesian representation.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Convert Cart3D to CartND:

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    >>> cxc.point_transition_map(cxc.cartnd, cxc.cart3d, p)
    {'q': Q([1., 2., 3.], 'm')}

    Convert Cart2D to CartND:

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> cxc.point_transition_map(cxc.cartnd, cxc.cart2d, p)
    {'q': Q([3., 4.], 'm')}

    Convert Radial to CartND:

    >>> p = {"r": u.Q(3.0, "m")}
    >>> cxc.point_transition_map(cxc.cartnd, cxc.radial1d, p)
    {'q': Q([3.], 'm')}

    Convert Cylindrical to CartND (z-axis point):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(5.0, "m")}
    >>> cxc.point_transition_map(cxc.cartnd, cxc.cyl3d, p)
    {'q': Q([0., 0., 5.], 'm')}

    """
    del to_chart  # Unused

    # If source is CartND, we can't convert (would be infinite recursion)
    if isinstance(from_chart, CartND):
        msg = "Cannot convert CartND to CartND via this dispatch."
        raise TypeError(msg)

    # Get the corresponding fixed-dimensional Cartesian chart
    cart_chart = from_chart.cartesian

    # If cartesian_chart returns CartND, we don't support this conversion
    if isinstance(cart_chart, CartND):
        msg = f"CartND conversion not supported for {type(from_chart).__name__}."
        raise NotImplementedError(msg)

    # Transform from source to fixed-dimensional Cartesian
    p_cart = api.point_transition_map(cart_chart, from_chart, p, usys=usys)

    # Convert fixed-dimensional Cartesian to CartND
    return _cartesian_to_cartnd(p_cart, cart_chart)


# ===================================================================
# SphericalTwoSphere <-> LonLatSphericalTwoSphere


@plum.dispatch
def point_transition_map(
    to_chart: LonLatSphericalTwoSphere,
    from_chart: SphericalTwoSphere,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """SphericalTwoSphere -> LonLatSphericalTwoSphere.

    lat = pi/2 - theta, lon = phi.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> p = {"theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}  # North pole
    >>> cxc.point_transition_map(cxc.lonlat_sph2, cxc.sph2, p)
    {'lon': Q(0, 'rad'), 'lat': Q(90., 'deg')}

    >>> p = {"theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}  # Equator
    >>> cxc.point_transition_map(cxc.lonlat_sph2, cxc.sph2, p)
    {'lon': Q(45, 'deg'), 'lat': Q(0., 'deg')}

    """
    del to_chart, from_chart  # unused
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    return {"lon": p["phi"], "lat": lat}


@plum.dispatch
def point_transition_map(
    to_chart: SphericalTwoSphere,
    from_chart: LonLatSphericalTwoSphere,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """LonLatSphericalTwoSphere -> SphericalTwoSphere.

    theta = pi/2 - lat, phi = lon.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"lon": u.Q(45, "deg"), "lat": u.Q(0, "deg")}
    >>> cxc.point_transition_map(cxc.sph2, cxc.lonlat_sph2, p)
    {'theta': Q(90., 'deg'), 'phi': Q(45, 'deg')}

    """
    del to_chart, from_chart  # unused
    theta = (
        u.Q(90, "deg") if isinstance(p["lat"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["lat"], usys)
    return {"theta": theta, "phi": p["lon"]}


# ===================================================================
# SphericalTwoSphere <-> LonCosLatSphericalTwoSphere


@plum.dispatch
def point_transition_map(
    to_chart: LonCosLatSphericalTwoSphere,
    from_chart: SphericalTwoSphere,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """SphericalTwoSphere -> LonCosLatSphericalTwoSphere.

    lat = pi/2 - theta, lon_coslat = phi * cos(lat).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> p = {"theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}  # equator
    >>> cxc.point_transition_map(cxc.loncoslat_sph2, cxc.sph2, p)
    {'lon_coslat': Q(45., 'deg'), 'lat': Q(0., 'deg')}

    >>> p = {"theta": u.Q(0, "deg"), "phi": u.Q(45, "deg")}  # north pole
    >>> result = cxc.point_transition_map(cxc.loncoslat_sph2, cxc.sph2, p)
    >>> bool(jnp.allclose(u.ustrip("deg", result["lat"]), 90.0))
    True

    """
    del to_chart, from_chart  # unused
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat)
    return {"lon_coslat": lon_coslat, "lat": lat}


@plum.dispatch
def point_transition_map(
    to_chart: SphericalTwoSphere,
    from_chart: LonCosLatSphericalTwoSphere,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """LonCosLatSphericalTwoSphere -> SphericalTwoSphere.

    theta = pi/2 - lat, phi = lon_coslat / cos(lat).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"lon_coslat": u.Q(45, "deg"), "lat": u.Q(0, "deg")}
    >>> cxc.point_transition_map(cxc.sph2, cxc.loncoslat_sph2, p)
    {'theta': Q(90., 'deg'), 'phi': Q(45., 'deg')}

    """
    del to_chart, from_chart  # unused
    lat = uconvert_to_rad(p["lat"], usys)
    theta = (u.Q(90, "deg") if isinstance(p["lat"], ABCQ) else jnp.pi / 2) - lat
    phi = p["lon_coslat"] / jnp.cos(lat)
    return {"theta": theta, "phi": phi}


# ===================================================================
# SphericalTwoSphere <-> MathSphericalTwoSphere


@plum.dispatch
def point_transition_map(
    to_chart: MathSphericalTwoSphere,
    from_chart: SphericalTwoSphere,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """SphericalTwoSphere -> MathSphericalTwoSphere.

    Swaps theta and phi (physics -> math convention).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
    >>> cxc.point_transition_map(cxc.math_sph2, cxc.sph2, p)
    {'theta': Q(60, 'deg'), 'phi': Q(30, 'deg')}

    """
    del to_chart, from_chart, usys  # Unused
    return {"theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def point_transition_map(
    to_chart: SphericalTwoSphere,
    from_chart: MathSphericalTwoSphere,
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """MathSphericalTwoSphere -> SphericalTwoSphere.

    Swaps theta and phi (math -> physics convention).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"theta": u.Q(60, "deg"), "phi": u.Q(30, "deg")}
    >>> cxc.point_transition_map(cxc.sph2, cxc.math_sph2, p)
    {'theta': Q(30, 'deg'), 'phi': Q(60, 'deg')}

    """
    del to_chart, from_chart, usys  # Unused
    return {"theta": p["phi"], "phi": p["theta"]}


# ===================================================================
# Point Transform a Quantity
# Only quantities which have the same units for all components can be
# transformed as a single Quantity.


@plum.dispatch.multi(
    *(
        (typ, typ, u.AbstractQuantity, OptUSys)
        for typ in (Abstract0D, Abstract1D, Cart2D, Cart3D, CartND)
    ),
    *(
        (typ, typ, u.AbstractQuantity)  # usys is optional
        for typ in (Abstract0D, Abstract1D, Cart2D, Cart3D, CartND)
    ),
)
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    q: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
) -> u.AbstractQuantity:
    """Identity point transform for Quantity inputs on uniform-unit charts.

    For charts where all components share the same unit (Cartesian charts,
    0D/1D charts), a Quantity can be passed directly and is returned unchanged
    when the source and target charts are the same type.

    This dispatch only handles identity transformations (same chart type).
    For transformations between different chart types with Quantity input,
    the Quantity must first be converted to a coordinate dictionary.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    **1D Cartesian (identity):**

    >>> q = u.Q([5.0], "m")
    >>> cxc.point_transition_map(cxc.cart1d, cxc.cart1d, q, None) is q
    True

    **2D Cartesian (identity):**

    >>> q = u.Q([3.0, 4.0], "m")
    >>> cxc.point_transition_map(cxc.cart2d, cxc.cart2d, q, None) is q
    True

    **3D Cartesian (identity):**

    >>> q = u.Q([1.0, 2.0, 3.0], "km")
    >>> cxc.point_transition_map(cxc.cart3d, cxc.cart3d, q, None) is q
    True

    **N-D Cartesian (identity):**

    >>> q = u.Q([1.0, 2.0, 3.0, 4.0], "m")
    >>> cxc.point_transition_map(cxc.cartnd, cxc.cartnd, q, None) is q
    True

    """
    del to_chart, from_chart, usys  # Unused
    return q


@plum.dispatch
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p: u.AbstractQuantity,
    /,
    *,
    usys: OptUSys = None,
) -> QuantityMatrix:
    """Transform a QuantityMatrix between charts.

    Converts the components of a QuantityMatrix from one chart to another,
    preserving the matrix structure with potentially different units per component.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    **2D Cartesian to Polar:**

    >>> q = u.Q([3.0, 4.0], "m")
    >>> result = cxc.point_transition_map(cxc.polar2d, cxc.cart2d, q)
    >>> result.shape
    (2,)
    >>> result.unit  # doctest: +SKIP
    UnitsMatrix((Unit("m"), Unit("rad")))

    **3D Cartesian to Spherical:**

    >>> q = u.Q([1.0, 0.0, 0.0], "kpc")
    >>> result = cxc.point_transition_map(cxc.sph3d, cxc.cart3d, q)
    >>> result.shape
    (3,)

    **Batched transformation:**

    >>> q_batch = u.Q([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "m")
    >>> result = cxc.point_transition_map(cxc.sph3d, cxc.cart3d, q_batch)
    >>> result.shape
    (2, 3)

    """
    # Build a dict of arrays for each component
    p_dict = api.cdict(from_chart, p)

    # Transform the point dict
    p_to = api.point_transition_map(to_chart, from_chart, p_dict, usys=usys)

    # Stack the transformed components into an QuantityMatrix
    p_out = QuantityMatrix(
        jnp.stack([u.ustrip(p_to[k]) for k in to_chart.components], axis=-1),
        unit=UnitsMatrix(u.unit_of(p_to[k]) for k in to_chart.components),
    )

    return p_out  # noqa: RET504


# ===================================================================
# Point Transform an Array


@plum.dispatch
def point_transition_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p: Array | list,  # type: ignore[type-arg]
    /,
    *,
    usys: OptUSys,
) -> Array:
    r"""Point transform for array input.

    Transforms a point represented as a raw array (without units) from one
    chart to another. The unit system ``usys`` provides the units for
    interpreting the array components.

    Returns
    -------
    Array
        Array of shape ``(..., ndim)`` containing the transformed coordinates
        in ``to_chart``.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    **Cartesian to Spherical (3D):**

    >>> usys = u.unitsystem("m", "rad")
    >>> p = jnp.array([1.0, 0.0, 0.0])  # Point on x-axis
    >>> cxc.point_transition_map(cxc.sph3d, cxc.cart3d, p, usys=usys)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    The result is [r, theta, phi] = [1, pi/2, 0] (on equator, x-axis).

    **Spherical to Cartesian (3D):**

    >>> p = jnp.array([2.0, jnp.pi/4, 0.0])  # r=2, theta=45°, phi=0
    >>> cxc.point_transition_map(cxc.cart3d, cxc.sph3d, p, usys=usys)
    Array([1.41421356, 0.        , 1.41421356], dtype=float64)

    **Cartesian to Cylindrical:**

    >>> p = jnp.array([3.0, 4.0, 5.0])
    >>> cxc.point_transition_map(cxc.cyl3d, cxc.cart3d, p, usys=usys)
    Array([5.        , 0.92729522, 5.        ], dtype=float64)

    The result is [rho, phi, z] = [5, arctan(4/3), 5].

    **Batched transformation:**

    >>> p_batch = jnp.array([[1.0, 0.0, 0.0],
    ...                      [0.0, 1.0, 0.0],
    ...                      [0.0, 0.0, 1.0]])
    >>> cxc.point_transition_map(cxc.sph3d, cxc.cart3d, p_batch, usys=usys)
    Array([[1.        , 1.57079633, 0.        ],
           [1.        , 1.57079633, 1.57079633],
           [1.        , 0.        , 0.        ]], dtype=float64)

    **2D Cartesian to Polar:**

    >>> usys_2d = u.unitsystem("m", "rad")
    >>> p = jnp.array([3.0, 4.0])
    >>> cxc.point_transition_map(cxc.polar2d, cxc.cart2d, p, usys=usys_2d)
    Array([5.        , 0.92729522], dtype=float64)

    """
    usys = eqx.error_if(usys, usys is None, "usys must be provided for array input.")

    # Build a dict of arrays for each component
    p_dict = api.cdict(from_chart, jnp.asarray(p))

    # Transform the point dict
    p_to = api.point_transition_map(to_chart, from_chart, p_dict, usys=usys)

    # Stack the transformed components into an array
    p_out: Array = jnp.stack([p_to[comp] for comp in to_chart.components], axis=-1)

    return p_out
