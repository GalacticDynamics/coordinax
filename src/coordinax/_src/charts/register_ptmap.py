"""Point-roled transformations in the same atlas."""

__all__: tuple[str, ...] = ()


from collections.abc import Callable
from jaxtyping import Array
from typing import Any, Final, cast, final

import jax
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as ABCQ  # noqa: N814

import coordinax.api.charts as cxcapi
from .d0 import Cart0D
from .d1 import Cart1D, Radial1D, Time1D
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
)
from .d6 import PoincarePolar6D
from .dn import CartND
from coordinax._src.base import AbstractChart
from coordinax._src.base.manifold import AbstractManifold
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.euclidean import RN, EuclideanManifold, Rn
from coordinax._src.utils import uconvert_to_rad
from coordinax.internal import QMatrix, UnitsMatrix, cdict_units


@final
class MissingType:
    """Sentinel for missing arguments."""


MISSING: Final[MissingType] = MissingType()


#####################################################################
# Point transformations

# ===================================================================
# Partial application


@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def pt_map(q: None, /, *fixed_args: Any, **fixed_kw: Any) -> Callable[..., Any]:
    """Return a partial function for point transformation.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Coordinates without units are the default.

    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d)
    >>> map(q)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    Coordinates without units are also accepted, interpreted having units of the
    `unxt.AbstractUnitSystem`, which must be passed.

    >>> q = {"x": 1.0, "y": 0.0, "z": 0.0}
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(q)
    {'r': Array(1., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64),
     'phi': Array(0., dtype=float64, ...)}

    `unxt.Quantity` inputs are also accepted, and are interpreted as being in
    Cartesian coordinates.

    >>> p = u.Q([1.0, 0.0, 0.0], "m")
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d)
    >>> map(p)
    QMatrix([1.        , 1.57079633, 0.        ], '(m, rad, rad)')

    Array-Like inputs are interpreted as Cartesian coordinates with units from
    the required `unxt.AbstractUnitSystem`.

    >>> q = [1.0, 0.0, 0.0]
    >>> map = cxc.pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(q)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    """
    del q  # unused

    # NOTE: lambda is much faster than ft.partial here
    return lambda x, *args, **kw: cxcapi.pt_map(x, *fixed_args, *args, **fixed_kw, **kw)


@plum.dispatch
def pt_map(
    from_chart: AbstractChart, to_chart: AbstractChart, /, **fixed_kw: Any
) -> Callable[..., Any]:
    """Return a partial function for point transformation.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Coordinates without units are the default.

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d)
    >>> map(p)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    Coordinates without units are also accepted, interpreted having units of the
    `unxt.AbstractUnitSystem`, which must be passed.

    >>> p = {"x": 1.0, "y": 0.0, "z": 0.0}
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(p)
    {'r': Array(1., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64),
     'phi': Array(0., dtype=float64, ...)}

    `unxt.Quantity` inputs are also accepted, and are interpreted as being in
    Cartesian coordinates.

    >>> p = u.Q([1.0, 0.0, 0.0], "m")
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d)
    >>> map(p)
    QMatrix([1.        , 1.57079633, 0.        ], '(m, rad, rad)')

    Array-Like inputs are interpreted as Cartesian coordinates with units from
    the required `unxt.AbstractUnitSystem`.

    >>> p = [1.0, 0.0, 0.0]
    >>> map = cxc.pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> map(p)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    """
    out = cxcapi.pt_map(None, from_chart, to_chart, **fixed_kw)
    return cast("Callable[..., Any]", out)


# ===================================================================
# Redispatch with the manifold


@plum.dispatch
def pt_map(
    x: Any,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Point transformation from chart to chart, using their manifolds.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {}
    >>> cxc.pt_map(p, cxc.cart0d, cxc.cart0d)
    {}

    >>> p = {"r": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxc.radial1d, cxc.cart1d)
    {'x': Q(5., 'm')}

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxc.pt_map(p, cxc.polar2d, cxc.cart2d)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxc.sph3d, cxc.cart3d)
    {'x': Q(5., 'm'), 'y': Q(0., 'm'), 'z': Q(3.061617e-16, 'm')}

    """
    return cxcapi.pt_map(x, from_chart.M, from_chart, to_chart.M, to_chart, usys=usys)


# ===================================================================
# General representation conversions


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def pt_map(
    p: CDict,
    from_M: AbstractManifold,
    from_chart: AbstractChart,
    to_M: AbstractManifold,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> Cartesian -> AbstractChart.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> map = cxc.pt_map.invoke(dict[str, u.Q], cxm.Rn, cxc.AbstractChart,
    ...                         cxm.Rn, cxc.AbstractChart)
    >>> map(p, cxm.R2, cxc.polar2d, cxm.R2, cxc.cart2d)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    # Even though there's a dispatch for the Self-to-Self case, we still check
    # for it here to avoid infinite recursion.
    if from_chart == to_chart:
        return p

    # Now we know from_chart and to_chart are different, so we can safely call.
    from_cart = from_chart.cartesian
    to_cart = to_chart.cartesian
    assert from_cart == to_cart  # noqa: S101

    p_cart = cxcapi.pt_map(p, from_M, from_chart, to_M, from_cart, usys=usys)
    p_out = cxcapi.pt_map(p_cart, from_M, from_cart, to_M, to_chart, usys=usys)
    return cast("CDict", p_out)


##############################################################################
# Transition Maps Assuming on Same Atlas

# ===================================================================
# Self representation conversions

IDENTITY_TRANSFORM_CHARTS: Final[tuple[type[AbstractChart[Any, Any, Any]], ...]] = (
    # 0D
    Cart0D,
    # 1D
    Cart1D,
    Radial1D,
    Time1D,
    # 2D
    Cart2D,
    Polar2D,
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
    # MinkowskiCT is registered separately via MinkowskiAtlas
)


@plum.dispatch.multi(*((CDict, Rn, typ, Rn, typ) for typ in IDENTITY_TRANSFORM_CHARTS))
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Identity conversion for matching charts.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> q = {}
    >>> q2 = cxc.pt_map(q, cxm.R0, cxc.cart0d, cxm.R0, cxc.cart0d)
    >>> q is q2
    True

    >>> q = {"r": u.Q(3.0, "m")}
    >>> q2 = cxc.pt_map(q, cxm.R1, cxc.radial1d, cxm.R1, cxc.radial1d)
    >>> q is q2
    True

    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> q2 = cxc.pt_map(q, cxm.R2, cxc.cart2d, cxm.R2, cxc.cart2d)
    >>> q is q2
    True

    """
    del usys  # unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    return p


# ---------------------------------------------------------
# Specific representation conversions

# -----------------------------------------------
# 1D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Radial1D,
    to_M: Rn,
    to_chart: Cart1D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Radial1D -> Cart1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> q = {"r": u.Q(5.0, "m")}
    >>> cxc.pt_map(q, cxm.R1, cxc.radial1d, cxm.R1, cxc.cart1d)
    {'x': Q(5., 'm')}

    >>> q = {"r": 5.0}  # No units
    >>> cxc.pt_map(q, cxm.R1, cxc.radial1d, cxm.R1, cxc.cart1d)
    {'x': 5.0}

    """
    del usys  # unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    return {"x": p["r"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart1D,
    to_M: Rn,
    to_chart: Radial1D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart1D -> Radial1D.

    The `x` coordinate is converted to the `r` coordinate of the 1D system.

    Assumptions:

    - Cart1D and Radial1D are

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxm.R1, cxc.cart1d, cxm.R1, cxc.radial1d)
    {'r': Q(5., 'm')}

    >>> p = {"x": 5.0}  # No units
    >>> cxc.pt_map(p, cxm.R1, cxc.cart1d, cxm.R1, cxc.radial1d)
    {'r': 5.0}

    """
    del usys  # unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    return {"r": p["x"]}


# -----------------------------------------------
# 2D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Polar2D,
    to_M: Rn,
    to_chart: Cart2D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Polar2D -> Cart2D.

    The `r` and `theta` coordinates are converted to the `x` and `y` coordinates
    of the 2D Cartesian system.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxc.pt_map(p, cxm.R2, cxc.polar2d, cxm.R2, cxc.cart2d)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    >>> p = {"r": 5, "theta": 90}  # No units
    >>> usys = u.unitsystem("km", "deg")
    >>> cxc.pt_map(p, cxm.R2, cxc.polar2d, cxm.R2, cxc.cart2d, usys=usys)
    {'x': Array(3.061617e-16, dtype=float64, ...),
     'y': Array(5., dtype=float64, ...)}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    theta = uconvert_to_rad(p["theta"], usys)
    x = p["r"] * jnp.cos(theta)
    y = p["r"] * jnp.sin(theta)
    return {"x": x, "y": y}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart2D,
    to_M: Rn,
    to_chart: Polar2D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart2D -> Polar2D.

    The `x` and `y` coordinates are converted to the `r` and `theta` coordinates
    of the 2D polar system.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(3, "m"), "y": u.Q(4, "m")}
    >>> cxc.pt_map(p, cxm.R2, cxc.cart2d, cxm.R2, cxc.polar2d)
    {'r': Q(5., 'm'), 'theta': Q(0.92729522, 'rad')}

    >>> p = {"x": 3, "y": 4}  # No units
    >>> cxc.pt_map(p, cxm.R2, cxc.cart2d, cxm.R2, cxc.polar2d)
    {'r': Array(5., dtype=float64, ...),
     'theta': Array(0.92729522, dtype=float64, ...)}

    """
    del usys  # unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    r_ = jnp.hypot(p["x"], p["y"])
    theta = jnp.arctan2(p["y"], p["x"])
    return {"r": r_, "theta": theta}


# -----------------------------------------------
# 3D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cylindrical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cylindrical3D -> Cart3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"rho": u.Q(1.0, "m"), "phi": u.Q(90, "deg"), "z": u.Q(2.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.cart3d)
    {'x': Q(6.123234e-17, 'm'), 'y': Q(1., 'm'), 'z': Q(2., 'm')}

    >>> p = {"rho": 1.0, "phi": 90, "z": 2.0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.cart3d, usys=usys)
    {'x': Array(6.123234e-17, dtype=float64, ...), 'y': Array(1., dtype=float64, ...),
     'z': 2.0}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    phi = uconvert_to_rad(p["phi"], usys)
    x = p["rho"] * jnp.cos(phi)
    y = p["rho"] * jnp.sin(phi)
    return {"x": x, "y": y, "z": p["z"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> Cart3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(0., 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the equator (theta=90 deg, phi=0):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cart3d, usys=usys)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    r_ = p["r"]
    theta = uconvert_to_rad(p["theta"], usys)
    phi = uconvert_to_rad(p["phi"], usys)
    x = r_ * jnp.sin(theta) * jnp.cos(phi)
    y = r_ * jnp.sin(theta) * jnp.sin(phi)
    z = r_ * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: LonLatSpherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """LonLatSpherical3D -> Cart3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point at the north pole (lat=90 deg):

    >>> p = {"lon": u.Q(0, "deg"), "lat": u.Q(90, "deg"), "distance": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.lonlat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(6.123234e-17, 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the equator at lon=0:

    >>> p = {"lon": 0, "lat": 0, "distance": 2}
    >>> cxc.pt_map(p, cxm.R3, cxc.lonlat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Array(2., dtype=float64, ...),
     'y': Array(0., dtype=float64, ...),
     'z': Array(0., dtype=float64, ...)}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    r_ = p["distance"]
    lon = uconvert_to_rad(p["lon"], usys)
    lat = uconvert_to_rad(p["lat"], usys)
    x = r_ * jnp.cos(lat) * jnp.cos(lon)
    y = r_ * jnp.cos(lat) * jnp.sin(lon)
    z = r_ * jnp.sin(lat)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: LonCosLatSpherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """LonCosLatSpherical3D -> Cart3D.

    Components are (lon_coslat, lat, r), where lon_coslat := lon * cos(lat).
    Longitude is undefined at the poles (cos(lat) == 0); we set lon = 0 by
    convention there to avoid NaNs.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the equator (lat=0, so lon_coslat = lon):

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(0, "deg"),
    ...      "distance": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.loncoslat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(1., 'm'), 'y': Q(0., 'm'), 'z': Q(0., 'm')}

    At the north pole (lat=90), lon_coslat is effectively 0 regardless of lon:

    >>> p = {"lon_coslat": u.Q(0, "deg"), "lat": u.Q(90, "deg"),
    ...      "distance": u.Q(2.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.loncoslat_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(1.2246468e-16, 'm'), 'y': Q(0., 'm'), 'z': Q(2., 'm')}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

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
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: MathSpherical3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """MathSpherical3D -> Cart3D.

    - theta: azimuth in the x-y plane (longitude-like)
    - phi  : polar angle from +z, with phi in [0, pi]

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the +z axis (phi=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(0., 'm'), 'y': Q(0., 'm'), 'z': Q(1., 'm')}

    A point on the +x axis (theta=0, phi=90):

    >>> p = {"r": u.Q(2.0, "m"), "theta": u.Q(0, "deg"), "phi": u.Q(90, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.cart3d)
    {'x': Q(2., 'm'), 'y': Q(0., 'm'), 'z': Q(1.2246468e-16, 'm')}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    r_ = p["r"]
    theta = uconvert_to_rad(p["theta"], usys)
    phi = uconvert_to_rad(p["phi"], usys)
    x = r_ * jnp.sin(phi) * jnp.cos(theta)
    y = r_ * jnp.sin(phi) * jnp.sin(theta)
    z = r_ * jnp.cos(phi)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: ProlateSpheroidal3D,
    to_M: Rn,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""ProlateSpheroidal3D -> Cart3D.

    We calculate through cylindrical coordinates first:

    $\rho = \sqrt{(\mu-\Delta^2)\left(1-\frac{|\nu|}{\Delta^2}\right)}$
    $z = \sqrt{\mu\,\frac{|\nu|}{\Delta^2}}\;\mathrm{sign}(\nu)$
    $\phi = \phi.$

    Then convert to Cartesian:

    $x=\rho\cos\phi$, $y=\rho\sin\phi$, $z=z$.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cart3d)
    {'x': Q(0.8660254, 'm'), 'y': Q(0., 'm'), 'z': Q(1.11803399, 'm')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cart3d, usys=usys)
    {'x': Array(0.8660254, dtype=float64),
     'y': Array(0., dtype=float64),
     'z': Array(1.11803399, dtype=float64)}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    # Calculate cylindrical distance
    nu, mu = p["nu"], p["mu"]
    if not isinstance(nu, ABCQ) or not isinstance(mu, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'mu' or 'nu', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = cast("Array", u.ustrip(usys["length"], from_chart.Delta)) ** 2
    else:
        Delta2 = from_chart.Delta**2

    nu_D2 = jnp.abs(nu) / Delta2
    rho = jnp.sqrt((mu - Delta2) * (1 - nu_D2))

    # Convert to Cartesian
    phi = uconvert_to_rad(p["phi"], usys)
    x = rho * jnp.cos(phi)
    y = rho * jnp.sin(phi)
    z = jnp.sqrt(mu * nu_D2) * jnp.sign(nu)

    return {"x": x, "y": y, "z": z}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart3D,
    to_M: Rn,
    to_chart: Cylindrical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Cylindrical3D.

    >>> import coordinax.main as cx
    >>> import unxt as u

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.cyl3d)
    {'rho': Q(5., 'm'), 'phi': Q(0.92729522, 'rad'), 'z': Q(5., 'm')}

    >>> p = {"x": 3.0, "y": 4.0, "z": 5.0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.cyl3d)
    {'rho': Array(5., dtype=float64, ...),
     'phi': Array(0.92729522, dtype=float64, ...),
     'z': 5.0}

    """
    del usys  # Unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    rho = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    return {"rho": rho, "phi": phi, "z": p["z"]}


@plum.dispatch.multi(
    (CDict, EuclideanManifold, Cart3D, EuclideanManifold, AbstractSpherical3D),
    (CDict, EuclideanManifold, Cylindrical3D, EuclideanManifold, AbstractSpherical3D),
)
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart3D | Cylindrical3D,
    to_M: Rn,
    to_chart: AbstractSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Spherical3D -> AbstractSpherical3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.loncoslat_sph3d)
    {'lon_coslat': Q(0., 'rad'), 'lat': Q(90., 'deg'), 'distance': Q(1., 'm')}

    >>> p = {"rho": 0, "phi": 180, "z": 1}
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.loncoslat_sph3d, usys=usys)
    {'lon_coslat': Array(1.10218212e-14, dtype=float64),
     'lat': Array(1.57079633, dtype=float64),
     'distance': Array(1., dtype=float64, weak_type=True)}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    # from_chart -> Spherical3D -> to_chart
    sph3d = Spherical3D(M=from_chart.M)
    p_sph = cxcapi.pt_map(p, from_M, from_chart, to_M, sph3d, usys=usys)
    out = cxcapi.pt_map(p_sph, from_M, sph3d, to_M, to_chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cart3D,
    to_M: Rn,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cart3D -> Spherical3D.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the +z axis:

    >>> p = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Q(0., 'rad'), 'phi': Q(0., 'rad')}

    A point on the +x axis:

    >>> p = {"x": 2.0, "y": 0.0, "z": 0.0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.sph3d)
    {'r': Array(2., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64),
     'phi': Array(0., dtype=float64, ...)}

    """
    del usys
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    x, y, z = p["x"], p["y"], p["z"]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r == 0, jnp.ones(r.shape), z / r))
    # atan2 handles the case when x = y = 0, returning phi = 0
    phi = jnp.atan2(y, x)
    return {"r": r, "theta": theta, "phi": phi}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cylindrical3D,
    to_M: Rn,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Cylindrical3D -> Spherical3D.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(1.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Q(0., 'rad'), 'phi': Q(0, 'rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": 3.0, "phi": 0, "z": 0.0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, cxc.sph3d)
    {'r': Array(3., dtype=float64, ...), 'theta': Array(1.57079633, dtype=float64),
     'phi': 0}

    """
    del usys  # unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    r_ = jnp.hypot(p["rho"], p["z"])
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r_ == 0, jnp.ones(r_.shape), p["z"] / r_))
    return {"r": r_, "theta": theta, "phi": p["phi"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: Cylindrical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> Cylindrical3D.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    A point on the +z axis (theta=0):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cyl3d)
    {'rho': Q(0., 'm'), 'phi': Q(0, 'rad'), 'z': Q(1., 'm')}

    A point on the equator (theta=90 deg):

    >>> p = {"r": 2.0, "theta": 90, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.cyl3d, usys=usys)
    {'rho': Array(2., dtype=float64, ...), 'phi': 0,
     'z': Array(1.2246468e-16, dtype=float64, ...)}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    theta = uconvert_to_rad(p["theta"], usys)
    rho = p["r"] * jnp.sin(theta)
    z = p["r"] * jnp.cos(theta)
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: LonLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> LonLatSpherical3D.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Spherical theta=0 corresponds to lat=90 (north pole):

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.lonlat_sph3d)
    {'lon': Q(0, 'rad'), 'lat': Q(90., 'deg'), 'distance': Q(1., 'm')}

    Spherical theta=90 deg corresponds to lat=0 (equator):

    >>> p = {"r": 1.0, "theta": 0, "phi": 0}  # No units
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.lonlat_sph3d)
    {'lon': 0, 'lat': 1.5707963267948966, 'distance': 1.0}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    return {"lon": p["phi"], "lat": lat, "distance": p["r"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: LonCosLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> LonCosLatSpherical3D.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    On the equator (theta=90 deg), lon_coslat equals lon:

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.loncoslat_sph3d)
    {'lon_coslat': Q(45., 'deg'), 'lat': Q(0., 'deg'), 'distance': Q(1., 'm')}

    At the north pole (theta=0), lon_coslat = 0 regardless of phi:

    >>> p = {"r": 1.0, "theta": 0, "phi": 45}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.loncoslat_sph3d, usys=usys)
    {'lon_coslat': Array(2.7554553e-15, dtype=float64, ...),
     'lat': 1.5707963267948966, 'distance': 1.0}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat)
    return {"lon_coslat": lon_coslat, "lat": lat, "distance": p["r"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Spherical3D,
    to_M: Rn,
    to_chart: MathSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Spherical3D -> MathSpherical3D.

    Swaps theta and phi: Physics (theta=polar, phi=azimuth) to
    Math (theta=azimuth, phi=polar).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.math_sph3d)
    {'r': Q(1., 'm'), 'theta': Q(60, 'deg'), 'phi': Q(30, 'deg')}

    >>> p = {"r": 1.0, "theta": 30, "phi": 60}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.sph3d, cxm.R3, cxc.math_sph3d, usys=usys)
    {'r': 1.0, 'theta': 60, 'phi': 30}

    """
    del usys  # Unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: MathSpherical3D,
    to_M: Rn,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """MathSpherical3D -> Spherical3D.

    Swaps theta and phi: Math (theta=azimuth, phi=polar) to
    Physics (theta=polar, phi=azimuth).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(60, "deg"), "phi": u.Q(30, "deg")}
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Q(30, 'deg'), 'phi': Q(60, 'deg')}

    >>> p = {"r": 1.0, "theta": 60, "phi": 30}  # No units
    >>> usys = u.unitsystem("m", "deg")
    >>> cxc.pt_map(p, cxm.R3, cxc.math_sph3d, cxm.R3, cxc.sph3d, usys=usys)
    {'r': 1.0, 'theta': 30, 'phi': 60}

    """
    del usys  # Unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: ProlateSpheroidal3D,
    to_M: Rn,
    to_chart: Cylindrical3D,
    /,
    *,
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

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cyl3d)
    {'rho': Q(0.8660254, 'm'), 'phi': Q(0, 'rad'), 'z': Q(1.11803399, 'm')}

    >>> p = {"mu": 5.0, "nu": 1.0, "phi": 0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.pt_map(p, cxm.R3, prolatesph3d, cxm.R3, cxc.cyl3d, usys=usys)
    {'rho': Array(0.8660254, dtype=float64), 'phi': 0,
     'z': Array(1.11803399, dtype=float64)}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

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
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: Cylindrical3D,
    to_M: Rn,
    to_chart: ProlateSpheroidal3D,
    /,
    *,
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

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> prolatesph3d = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))

    A point on the z-axis (rho=0):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, prolatesph3d)
    {'mu': Q(9., 'm2'), 'nu': Q(4., 'm2'), 'phi': Q(0, 'rad')}

    A point in the xy-plane (z=0):

    >>> p = {"rho": u.Q(2.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(0.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, prolatesph3d)
    {'mu': Q(8., 'm2'), 'nu': Q(0., 'm2'), 'phi': Q(0, 'rad')}

    Without units:

    >>> p = {"rho": 2.0, "phi": 0, "z": 3.0}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.pt_map(p, cxm.R3, cxc.cyl3d, cxm.R3, prolatesph3d, usys=usys)
    {'mu': Array(14.52079729, dtype=float64),
     'nu': Array(2.47920271, dtype=float64), 'phi': 0}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    # Pre-compute common terms
    R2 = p["rho"] ** 2
    z2 = p["z"] ** 2
    if not isinstance(R2, ABCQ) or not isinstance(z2, ABCQ):
        if usys is None:
            msg = "For non-Quantity 'rho' or 'z', usys must be a UnitSystem, not None."
            raise ValueError(msg)

        Delta2 = cast("Array", u.ustrip(usys["length"], to_chart.Delta)) ** 2
    else:
        Delta2 = plum.convert(to_chart.Delta**2, u.Q)

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
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: ProlateSpheroidal3D,
    to_M: Rn,
    to_chart: ProlateSpheroidal3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""{class}`coordinax.charts.ProlateSpheroidal3D` -> itself.

    If the focal length is unchanged (``to_chart.Delta == from_chart.Delta``), this
    is the identity map.

    If the focal length changes, we convert via cylindrical coordinates:

    ``Prolate(Delta_in) -> Cylindrical -> Prolate(Delta_out)``.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Same focal length (identity transformation):

    >>> prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolate, cxm.R3, prolate)
    {'mu': Q(5., 'm2'),
     'nu': Q(1., 'm2'),
     'phi': Q(0., 'rad')}

    Different focal lengths (converts via cylindrical):

    >>> prolate_in = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    >>> prolate_out = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(3.0, "m"))
    >>> p = {"mu": u.Q(5.0, "m2"), "nu": u.Q(1.0, "m2"), "phi": u.Q(0, "rad")}
    >>> cxc.pt_map(p, cxm.R3, prolate_in, cxm.R3, prolate_out)
    {'mu': Q(9.85889894, 'm2'),
     'nu': Q(1.14110106, 'm2'),
     'phi': Q(0., 'rad')}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    # Cast to the result type
    dtype = jnp.result_type(
        to_chart.Delta, from_chart.Delta, *[v.dtype for v in p.values()]
    )
    p = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), p)
    cyl3d = Cylindrical3D(M=to_chart.M)
    return jax.lax.cond(
        to_chart.Delta == from_chart.Delta,
        lambda p: p,
        lambda p: cxcapi.pt_map(
            cxcapi.pt_map(p, from_M, from_chart, to_M, cyl3d, usys=usys),
            from_M,
            cyl3d,
            to_M,
            to_chart,
            usys=usys,
        ),
        p,
    )


# -----------------------------------------------
# N-D


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: CartND,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """CartND -> AbstractChart.

    Converts from N-dimensional Cartesian (with a single 'q' array) to any
    other chart type by first extracting the appropriate fixed-dimensional
    Cartesian representation.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Convert 3D CartND to Spherical:

    >>> p = {"q": u.Q([1.0, 0.0, 0.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R3, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    Convert 2D CartND to Polar:

    >>> p = {"q": u.Q([3.0, 4.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R2, cxc.polar2d)
    {'r': Q(5., 'm'), 'theta': Q(0.92729522, 'rad')}

    Convert 1D CartND to Radial:

    >>> p = {"q": u.Q([5.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R1, cxc.radial1d)
    {'r': Q(5., 'm')}

    Convert CartND to Cart3D:

    >>> p = {"q": u.Q([1.0, 2.0, 3.0], "m")}
    >>> cxc.pt_map(p, cxm.RN, cxc.cartnd, cxm.R3, cxc.cart3d)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    assert from_chart.M in (from_M, to_chart.M)  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

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
    if data_ndim != target_ndim:
        msg = (
            f"CartND data has {data_ndim} dimensions but target chart "
            f"{type(to_chart).__name__} requires {target_ndim} dimensions."
        )
        raise ValueError(msg)

    # Convert CartND to fixed-dimensional Cartesian
    p_cart = {k: q[..., i] for i, k in enumerate(cart_chart.components)}

    # If target is already the Cartesian chart, return directly
    if type(to_chart) is type(cart_chart):
        return p_cart

    # Otherwise, transform from Cartesian to target chart
    out = cxcapi.pt_map(p_cart, cart_chart.M, cart_chart, to_M, to_chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: CartND,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """AbstractChart -> CartND.

    Converts from any chart type to N-dimensional Cartesian (with a single
    'q' array) by first transforming to the appropriate fixed-dimensional
    Cartesian representation.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Convert Cart3D to CartND:

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxm.R3, cxc.cart3d, cxm.R3, cxc.cartnd)
    {'q': Q([1., 2., 3.], 'm')}

    Convert Cart2D to CartND:

    >>> p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
    >>> cxc.pt_map(p, cxm.R2, cxc.cart2d, cxm.R2, cxc.cartnd)
    {'q': Q([3., 4.], 'm')}

    Convert Radial to CartND:

    >>> p = {"r": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxc.radial1d, cxc.cartnd)
    {'q': Q([3.], 'm')}

    Convert Cylindrical to CartND (z-axis point):

    >>> p = {"rho": u.Q(0.0, "m"), "phi": u.Q(0, "rad"), "z": u.Q(5.0, "m")}
    >>> cxc.pt_map(p, cxc.cyl3d, cxc.cartnd)
    {'q': Q([0., 0., 5.], 'm')}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_chart.M in (from_M, RN)  # noqa: S101
    assert to_M in (from_M, RN)  # noqa: S101

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
    p_cart = cxcapi.pt_map(p, from_M, from_chart, from_M, cart_chart, usys=usys)
    p_cart = cast("dict[str, Array]", p_cart)

    # Convert fixed-dimensional Cartesian to CartND
    q = jnp.stack([p_cart[k] for k in cart_chart.components], axis=-1)

    return {"q": q}


# ===================================================================
# Point Transform a Quantity
# Only quantities which have the same units for all components can be
# transformed as a single Quantity.


@plum.dispatch.multi(
    *(
        (u.AbstractQuantity, EuclideanManifold, typ, EuclideanManifold, typ)
        for typ in (Cart0D, Cart1D, Radial1D, Time1D, Cart2D, Cart3D, CartND)
    ),
)
def pt_map(
    q: u.AbstractQuantity,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> u.AbstractQuantity:
    """Identity point transform for Quantity inputs on uniform-unit charts.

    For charts where all components share the same unit (Cartesian charts, 0D/1D
    charts), a Quantity can be passed directly and is returned unchanged when
    the source and target charts are the same type.

    This dispatch only handles identity transformations (same chart type).  For
    transformations between different chart types with Quantity input, the
    Quantity must first be converted to a coordinate dictionary.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    **1D Cartesian (identity):**

    >>> q = u.Q([5.0], "m")
    >>> cxc.pt_map(q, cxm.R1, cxc.cart1d, cxm.R1, cxc.cart1d, usys=None) is q
    True

    **2D Cartesian (identity):**

    >>> q = u.Q([3.0, 4.0], "m")
    >>> cxc.pt_map(q, cxm.R2, cxc.cart2d, cxm.R2, cxc.cart2d, usys=None) is q
    True

    **3D Cartesian (identity):**

    >>> q = u.Q([1.0, 2.0, 3.0], "km")
    >>> cxc.pt_map(q, cxm.R3, cxc.cart3d, cxm.R3, cxc.cart3d, usys=None) is q
    True

    **N-D Cartesian (identity):**

    >>> q = u.Q([1.0, 2.0, 3.0, 4.0], "m")
    >>> cxc.pt_map(q, cxm.RN, cxc.cartnd, cxm.RN, cxc.cartnd, usys=None) is q
    True

    """
    del usys  # Unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    return q


@plum.dispatch
def pt_map(
    p: u.AbstractQuantity,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> QMatrix:
    """Transform a QMatrix between charts.

    Converts the components of a QMatrix from one chart to another,
    preserving the matrix structure with potentially different units per component.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    **2D Cartesian to Polar:**

    >>> q = u.Q([3.0, 4.0], "m")
    >>> result = cxc.pt_map(q, cxc.cart2d, cxc.polar2d)
    >>> result.shape
    (2,)
    >>> result.unit
    UnitsMatrix("(m, rad)")

    **3D Cartesian to Spherical:**

    >>> q = u.Q([1.0, 0.0, 0.0], "kpc")
    >>> result = cxc.pt_map(q, cxc.cart3d, cxc.sph3d)
    >>> result.shape
    (3,)

    **Batched transformation:**

    >>> q_batch = u.Q([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "m")
    >>> result = cxc.pt_map(q_batch, cxc.cart3d, cxc.sph3d)
    >>> result.shape
    (2, 3)

    """
    # Build a dict of arrays for each component
    p_dict = cxcapi.cdict(p, from_chart)

    # Transform the point dict
    p_to = cxcapi.pt_map(p_dict, from_M, from_chart, to_M, to_chart, usys=usys)
    p_to = cast("dict[str, u.AbstractQuantity]", p_to)

    # Stack the transformed components into an QMatrix
    p_out = QMatrix(
        jnp.stack([u.ustrip(p_to[k]) for k in to_chart.components], axis=-1),
        unit=UnitsMatrix(cdict_units(p_to, to_chart.components)),
    )

    return p_out  # noqa: RET504


# ===================================================================
# Point Transform an Array


@plum.dispatch
def pt_map(
    p: Array | list,
    from_M: Rn,
    from_chart: AbstractChart,
    to_M: Rn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys,
) -> Array:
    r"""Point transform for array input.

    Transforms a point represented as a raw array (without units) from one chart
    to another. The unit system ``usys`` provides the units for interpreting the
    array components.

    Returns
    -------
    Array
        Array of shape ``(..., ndim)`` containing the transformed coordinates in
        ``to_chart``.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    **Cartesian to Spherical (3D):**

    >>> usys = u.unitsystem("m", "rad")
    >>> p = jnp.array([1.0, 0.0, 0.0])  # Point on x-axis
    >>> cxc.pt_map(p, cxc.cart3d, cxc.sph3d, usys=usys)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    The result is [r, theta, phi] = [1, pi/2, 0] (on equator, x-axis).

    **Spherical to Cartesian (3D):**

    >>> p = jnp.array([2.0, jnp.pi/4, 0.0])  # r=2, theta=45°, phi=0
    >>> cxc.pt_map(p, cxc.sph3d, cxc.cart3d, usys=usys)
    Array([1.41421356, 0.        , 1.41421356], dtype=float64)

    **Cartesian to Cylindrical:**

    >>> p = jnp.array([3.0, 4.0, 5.0])
    >>> cxc.pt_map(p, cxc.cart3d, cxc.cyl3d, usys=usys)
    Array([5.        , 0.92729522, 5.        ], dtype=float64)

    The result is [rho, phi, z] = [5, arctan(4/3), 5].

    **Batched transformation:**

    >>> p_batch = jnp.array([[1.0, 0.0, 0.0],
    ...                      [0.0, 1.0, 0.0],
    ...                      [0.0, 0.0, 1.0]])
    >>> cxc.pt_map(p_batch, cxc.cart3d, cxc.sph3d, usys=usys)
    Array([[1.        , 1.57079633, 0.        ],
           [1.        , 1.57079633, 1.57079633],
           [1.        , 0.        , 0.        ]], dtype=float64)

    **2D Cartesian to Polar:**

    >>> usys_2d = u.unitsystem("m", "rad")
    >>> p = jnp.array([3.0, 4.0])
    >>> cxc.pt_map(p, cxc.cart2d, cxc.polar2d, usys=usys_2d)
    Array([5.        , 0.92729522], dtype=float64)

    """
    if usys is None:
        msg = "usys must be provided for array input."
        raise ValueError(msg)

    # Build a dict of arrays for each component
    p_dict = cxcapi.cdict(jnp.asarray(p), from_chart)

    # Transform the point dict
    p_to = cxcapi.pt_map(p_dict, from_M, from_chart, to_M, to_chart, usys=usys)
    p_to = cast("dict[str, Array]", p_to)

    # Stack the transformed components into an array
    p_out: Array = jnp.stack([p_to[comp] for comp in to_chart.components], axis=-1)

    return p_out
