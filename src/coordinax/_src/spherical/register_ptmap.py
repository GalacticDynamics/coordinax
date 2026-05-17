"""Point-roled transformations in the same atlas."""

__all__: tuple[str, ...] = ()


from typing import Any, Final

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity

from .chart import (
    LonCosLatSphericalTwoSphere,
    LonLatSphericalTwoSphere,
    MathSphericalTwoSphere,
    SphericalTwoSphere,
)
from .manifold import Sn
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.utils import uconvert_to_rad

IDENTITY_TRANSFORM_CHARTS: Final[tuple[type[AbstractChart[Any, Any]], ...]] = (
    SphericalTwoSphere,
    LonLatSphericalTwoSphere,
    LonCosLatSphericalTwoSphere,
    MathSphericalTwoSphere,
)


@plum.dispatch.multi(*((CDict, Sn, typ, Sn, typ) for typ in IDENTITY_TRANSFORM_CHARTS))
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: AbstractChart,
    to_M: Sn,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Identity conversion for matching charts.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> q = {"theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
    >>> cxc.pt_map(q, cxc.sph2, cxc.sph2) is q
    True

    >>> q = {"lon": u.Q(45, "deg"), "lat": u.Q(10, "deg")}
    >>> cxc.pt_map(q, cxc.lonlat_sph2, cxc.lonlat_sph2) is q
    True

    >>> q = {"lon_coslat": u.Q(30, "deg"), "lat": u.Q(20, "deg")}
    >>> cxc.pt_map(q, cxc.loncoslat_sph2, cxc.loncoslat_sph2) is q
    True

    >>> q = {"theta": u.Q(60, "deg"), "phi": u.Q(30, "deg")}
    >>> cxc.pt_map(q, cxc.math_sph2, cxc.math_sph2) is q
    True

    """
    del usys  # unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    return p


# ===================================================================
# SphericalTwoSphere <-> LonLatSphericalTwoSphere


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: SphericalTwoSphere,
    to_M: Sn,
    to_chart: LonLatSphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """SphericalTwoSphere -> LonLatSphericalTwoSphere.

    lat = pi/2 - theta, lon = phi.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}  # North pole
    >>> cxc.pt_map(p, cxm.S2, cxc.sph2, cxm.S2, cxc.lonlat_sph2)
    {'lon': Q(0, 'rad'), 'lat': Q(90., 'deg')}

    >>> p = {"theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}  # Equator
    >>> cxc.pt_map(p, cxm.S2, cxc.sph2, cxm.S2, cxc.lonlat_sph2)
    {'lon': Q(45, 'deg'), 'lat': Q(0, 'deg')}

    """
    del usys  # Unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    lat = p["theta"]
    lat = u.Q(90, "deg") - lat if is_any_quantity(lat) else jnp.pi / 2 - lat
    return {"lon": p["phi"], "lat": lat}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: LonLatSphericalTwoSphere,
    to_M: Sn,
    to_chart: SphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """LonLatSphericalTwoSphere -> SphericalTwoSphere.

    theta = pi/2 - lat, phi = lon.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"lon": u.Q(45, "deg"), "lat": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxm.S2, cxc.lonlat_sph2, cxm.S2, cxc.sph2)
    {'theta': Q(90, 'deg'), 'phi': Q(45, 'deg')}

    """
    del usys
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    theta = p["lat"]
    theta = u.Q(90, "deg") - theta if is_any_quantity(theta) else jnp.pi / 2 - theta
    return {"theta": theta, "phi": p["lon"]}


# ===================================================================
# SphericalTwoSphere <-> LonCosLatSphericalTwoSphere


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: SphericalTwoSphere,
    to_M: Sn,
    to_chart: LonCosLatSphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """SphericalTwoSphere -> LonCosLatSphericalTwoSphere.

    lat = pi/2 - theta, lon_coslat = phi * cos(lat).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> p = {"theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}  # equator
    >>> cxc.pt_map(p, cxm.S2, cxc.sph2, cxm.S2, cxc.loncoslat_sph2)
    {'lon_coslat': Q(45., 'deg'), 'lat': Q(0., 'deg')}

    >>> p = {"theta": u.Q(0, "deg"), "phi": u.Q(45, "deg")}  # north pole
    >>> result = cxc.pt_map(p, cxm.S2, cxc.sph2, cxm.S2, cxc.loncoslat_sph2)
    >>> bool(jnp.allclose(u.ustrip("deg", result["lat"]), 90.0))
    True

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    lat = (
        u.Q(90, "deg") if is_any_quantity(p["theta"]) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat)
    return {"lon_coslat": lon_coslat, "lat": lat}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: LonCosLatSphericalTwoSphere,
    to_M: Sn,
    to_chart: SphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """LonCosLatSphericalTwoSphere -> SphericalTwoSphere.

    theta = pi/2 - lat, phi = lon_coslat / cos(lat).

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    >>> p = {"lon_coslat": u.Q(45, "deg"), "lat": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxm.S2, cxc.loncoslat_sph2, cxm.S2, cxc.sph2)
    {'theta': Q(90., 'deg'), 'phi': Q(45., 'deg')}

    """
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    lat = uconvert_to_rad(p["lat"], usys)
    theta = (u.Q(90, "deg") if is_any_quantity(p["lat"]) else jnp.pi / 2) - lat
    phi = p["lon_coslat"] / jnp.cos(lat)
    return {"theta": theta, "phi": phi}


# ===================================================================
# SphericalTwoSphere <-> MathSphericalTwoSphere


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: SphericalTwoSphere,
    to_M: Sn,
    to_chart: MathSphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """SphericalTwoSphere -> MathSphericalTwoSphere.

    Swaps theta and phi (physics -> math convention).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"theta": u.Q(30, "deg"), "phi": u.Q(60, "deg")}
    >>> cxc.pt_map(p, cxm.S2, cxc.sph2, cxm.S2, cxc.math_sph2)
    {'theta': Q(60, 'deg'), 'phi': Q(30, 'deg')}

    """
    del usys  # Unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101
    return {"theta": p["phi"], "phi": p["theta"]}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: MathSphericalTwoSphere,
    to_M: Sn,
    to_chart: SphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """MathSphericalTwoSphere -> SphericalTwoSphere.

    Swaps theta and phi (math -> physics convention).

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p = {"theta": u.Q(60, "deg"), "phi": u.Q(30, "deg")}
    >>> cxc.pt_map(p, cxm.S2, cxc.math_sph2, cxm.S2, cxc.sph2)
    {'theta': Q(30, 'deg'), 'phi': Q(60, 'deg')}

    """
    del usys  # Unused
    assert from_M == from_chart.M  # noqa: S101
    assert to_M == to_chart.M  # noqa: S101

    return {"theta": p["phi"], "phi": p["theta"]}
