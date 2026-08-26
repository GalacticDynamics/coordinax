"""Point-roled transformations in the same atlas."""

__all__: tuple[str, ...] = ()


from typing import Any, Final, cast

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity

import coordinaxs.api.charts as cxcapi
from .chart import (
    AbstractSphericalTwoSphere,
    LonCosLatSphericalTwoSphere,
    LonLatSphericalTwoSphere,
    MathSphericalTwoSphere,
    SphericalTwoSphere,
)
from .manifold import Sn
from coordinax._src.base import AbstractChart
from coordinax._src.charts.checks import check_manifolds_match_charts
from coordinax._src.charts.containers import canonical_containers
from coordinax._src.custom_types import OptUSys
from coordinax._src.utils import uconvert_to_rad
from coordinaxs.api.custom_types import CDict

IDENTITY_TRANSFORM_CHARTS: Final[tuple[type[AbstractChart[Any, Any, Any]], ...]] = (
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

    Returns the input object itself when it is already canonical -- angular
    components held as `unxt.Angle`, as the chart declares. A non-canonical
    input is canonicalised instead, so it necessarily comes back as a new dict:
    the alternative is `pt_map(q, chart, chart)` preserving a container that
    every other route to the same chart would have normalised.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> q = {"theta": u.Angle(30, "deg"), "phi": u.Angle(60, "deg")}
    >>> cxc.pt_map(q, cxc.sph2, cxc.sph2) is q
    True

    >>> q = {"lon": u.Angle(45, "deg"), "lat": u.Angle(10, "deg")}
    >>> cxc.pt_map(q, cxc.lonlat_sph2, cxc.lonlat_sph2) is q
    True

    >>> q = {"lon_coslat": u.Angle(30, "deg"), "lat": u.Angle(20, "deg")}
    >>> cxc.pt_map(q, cxc.loncoslat_sph2, cxc.loncoslat_sph2) is q
    True

    >>> q = {"theta": u.Angle(60, "deg"), "phi": u.Angle(30, "deg")}
    >>> cxc.pt_map(q, cxc.math_sph2, cxc.math_sph2) is q
    True

    """
    del usys  # unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    return canonical_containers(p, to_chart)


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Sn,
    from_chart: AbstractSphericalTwoSphere,
    to_M: Sn,
    to_chart: AbstractSphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Route between two-sphere charts via `SphericalTwoSphere`.

    Each chart registers only its direct ``sph2 <-> chart`` conversion, and the
    two-sphere has no Cartesian chart, so the generic router cannot bridge two
    non-canonical charts (it raises ``NoGlobalCartesianChartError``). Go
    ``A -> SphericalTwoSphere -> B`` instead. Canonical pairs (either side is
    `SphericalTwoSphere`) and matching-type pairs are handled by the more
    specific direct/identity rules above, so this fallback fires only for
    distinct non-canonical charts -- and covers any future
    `AbstractSphericalTwoSphere` subclass automatically.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    >>> p = {"lon": u.Q(45, "deg"), "lat": u.Q(30, "deg")}
    >>> out = cxc.pt_map(p, cxm.S2, cxc.lonlat_sph2, cxm.S2, cxc.math_sph2)
    >>> sorted(out)
    ['phi', 'theta']

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    canon = SphericalTwoSphere(M=to_M)
    p_canon = cxcapi.pt_map(p, from_M, from_chart, to_M, canon, usys=usys)
    out = cxcapi.pt_map(p_canon, to_M, canon, to_M, to_chart, usys=usys)
    return canonical_containers(cast("CDict", out), to_chart)


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
    {'lon': Angle(0, 'rad'), 'lat': Angle(90., 'deg')}

    >>> p = {"theta": u.Q(90, "deg"), "phi": u.Q(45, "deg")}  # Equator
    >>> cxc.pt_map(p, cxm.S2, cxc.sph2, cxm.S2, cxc.lonlat_sph2)
    {'lon': Angle(45, 'deg'), 'lat': Angle(0, 'deg')}

    """
    del usys  # Unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    lat = p["theta"]
    lat = u.Q(90, "deg") - lat if is_any_quantity(lat) else jnp.pi / 2 - lat
    return canonical_containers({"lon": p["phi"], "lat": lat}, to_chart)


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
    {'theta': Angle(90, 'deg'), 'phi': Angle(45, 'deg')}

    """
    del usys
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    theta = p["lat"]
    theta = u.Q(90, "deg") - theta if is_any_quantity(theta) else jnp.pi / 2 - theta
    return canonical_containers({"theta": theta, "phi": p["lon"]}, to_chart)


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
    {'lon_coslat': Angle(45., 'deg'), 'lat': Angle(0., 'deg')}

    >>> p = {"theta": u.Q(0, "deg"), "phi": u.Q(45, "deg")}  # north pole
    >>> result = cxc.pt_map(p, cxm.S2, cxc.sph2, cxm.S2, cxc.loncoslat_sph2)
    >>> bool(jnp.allclose(u.ustrip("deg", result["lat"]), 90.0))
    True

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    lat = (
        u.Q(90, "deg") if is_any_quantity(p["theta"]) else jnp.pi / 2
    ) - uconvert_to_rad(p["theta"], usys)
    lon_coslat = p["phi"] * jnp.cos(lat)
    return canonical_containers({"lon_coslat": lon_coslat, "lat": lat}, to_chart)


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
    {'theta': Angle(90., 'deg'), 'phi': Angle(45., 'deg')}

    """
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    lat = uconvert_to_rad(p["lat"], usys)
    theta = (u.Q(90, "deg") if is_any_quantity(p["lat"]) else jnp.pi / 2) - lat
    phi = p["lon_coslat"] / jnp.cos(lat)
    return canonical_containers({"theta": theta, "phi": phi}, to_chart)


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
    {'theta': Angle(60, 'deg'), 'phi': Angle(30, 'deg')}

    """
    del usys  # Unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)
    return canonical_containers({"theta": p["phi"], "phi": p["theta"]}, to_chart)


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
    {'theta': Angle(30, 'deg'), 'phi': Angle(60, 'deg')}

    """
    del usys  # Unused
    check_manifolds_match_charts(from_M, from_chart, to_M, to_chart)

    return canonical_containers({"theta": p["phi"], "phi": p["theta"]}, to_chart)
