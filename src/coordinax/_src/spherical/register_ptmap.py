"""Point-roled transformations in the same atlas."""

__all__: tuple[str, ...] = ()


from typing import Any, Final

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as ABCQ  # noqa: N814
from unxt.quantity import is_any_quantity

from .chart import (
    LonCosLatSphericalTwoSphere,
    LonLatSphericalTwoSphere,
    MathSphericalTwoSphere,
    SphericalTwoSphere,
)
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.utils import uconvert_to_rad

IDENTITY_TRANSFORM_CHARTS: Final[tuple[type[AbstractChart[Any, Any]], ...]] = (
    SphericalTwoSphere,
    LonLatSphericalTwoSphere,
    LonCosLatSphericalTwoSphere,
    MathSphericalTwoSphere,
)


@plum.dispatch.multi(*((CDict, typ, typ) for typ in IDENTITY_TRANSFORM_CHARTS))
def pt_map(
    p: CDict,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Identity conversion for matching charts."""
    del from_chart, to_chart, usys  # unused
    return p


# ===================================================================
# SphericalTwoSphere <-> LonLatSphericalTwoSphere


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: SphericalTwoSphere,
    to_chart: LonLatSphericalTwoSphere,
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
    >>> cxc.pt_map(p, cxc.sph2, cxc.lonlat_sph2)
    {'lon': Q(0, 'rad'), 'lat': Q(90., 'deg')}

    >>> p = {"theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}  # Equator
    >>> cxc.pt_map(p, cxc.sph2, cxc.lonlat_sph2)
    {'lon': Q(45, 'deg'), 'lat': Q(0., 'deg')}

    """
    del to_chart, from_chart  # unused
    lat = (
        u.Q(90, "deg") if isinstance(p["theta"], ABCQ) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    return {"lon": p["phi"], "lat": lat}


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: LonLatSphericalTwoSphere,
    to_chart: SphericalTwoSphere,
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
    >>> cxc.pt_map(p, cxc.lonlat_sph2, cxc.sph2)
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
def pt_map(
    p: CDict,
    from_chart: SphericalTwoSphere,
    to_chart: LonCosLatSphericalTwoSphere,
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
    >>> cxc.pt_map(p, cxc.sph2, cxc.loncoslat_sph2)
    {'lon_coslat': Q(45., 'deg'), 'lat': Q(0., 'deg')}

    >>> p = {"theta": u.Q(0, "deg"), "phi": u.Q(45, "deg")}  # north pole
    >>> result = cxc.pt_map(p, cxc.sph2, cxc.loncoslat_sph2)
    >>> bool(jnp.allclose(u.ustrip("deg", result["lat"]), 90.0))
    True

    """
    del to_chart, from_chart  # unused
    lat = (
        u.Q(90, "deg") if is_any_quantity(p["theta"]) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat)
    return {"lon_coslat": lon_coslat, "lat": lat}


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: LonCosLatSphericalTwoSphere,
    to_chart: SphericalTwoSphere,
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
    >>> cxc.pt_map(p, cxc.loncoslat_sph2, cxc.sph2)
    {'theta': Q(90., 'deg'), 'phi': Q(45., 'deg')}

    """
    del to_chart, from_chart  # unused
    lat = uconvert_to_rad(p["lat"], usys)
    theta = (u.Q(90, "deg") if is_any_quantity(p["lat"]) else jnp.pi / 2) - lat
    phi = p["lon_coslat"] / jnp.cos(lat)
    return {"theta": theta, "phi": phi}


# ===================================================================
# SphericalTwoSphere <-> MathSphericalTwoSphere


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: SphericalTwoSphere,
    to_chart: MathSphericalTwoSphere,
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
    >>> cxc.pt_map(p, cxc.sph2, cxc.math_sph2)
    {'theta': Q(60, 'deg'), 'phi': Q(30, 'deg')}

    """
    del to_chart, from_chart, usys  # Unused
    return {"theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: MathSphericalTwoSphere,
    to_chart: SphericalTwoSphere,
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
    >>> cxc.pt_map(p, cxc.math_sph2, cxc.sph2)
    {'theta': Q(30, 'deg'), 'phi': Q(60, 'deg')}

    """
    del to_chart, from_chart, usys  # Unused
    return {"theta": p["phi"], "phi": p["theta"]}
