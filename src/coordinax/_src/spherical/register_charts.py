"""Manifold definitions and manifold inference helpers."""

__all__: tuple[str, ...] = ()


from typing import Any, Final, cast

import plum

import coordinax.api.charts as cxcapi
import coordinax.api.manifolds as cxmapi
from .chart import AbstractSphericalTwoSphere
from .embed import TwoSphereIn3D
from .manifold import HyperSphericalManifold
from coordinax._src.base import AbstractChart
from coordinax._src.charts.d3 import Abstract3D, Spherical3D
from coordinax._src.custom_types import CDict, OptUSys

_twospherefrom3d: Final = TwoSphereIn3D(1)


@plum.dispatch
def pt_project(
    p_ambient: object,
    from_ambient_chart: AbstractChart,
    M: HyperSphericalManifold,
    /,
    *,
    usys: OptUSys = None,
) -> object:
    """Project a point from the 3D chart to the two-sphere intrinsic chart.

    This projection map is a special case for projecting from 3D charts to the
    two-sphere intrinsic chart, which is a common use case. The projection does
    not depend on the radius of the embedding, so this projection works in
    general.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> q = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
    >>> M = cxm.HyperSphericalManifold()
    >>> cxm.pt_project(q, cxc.cart3d, M)
    {'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    """
    del M

    # First project from the ambient chart to the intermediate Spherical3D chart
    sph3d = Spherical3D(M=from_ambient_chart.M)
    x_sph = cxcapi.pt_map(p_ambient, from_ambient_chart, sph3d)

    # Then project from the intermediate Spherical3D chart to the intrinsic
    # SphericalTwoSphere chart. The radius doesn't matter for the projection.
    return _twospherefrom3d.project(x_sph, usys=usys)


@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def pt_map(
    p: Any,
    from_chart: Abstract3D,
    to_chart: AbstractSphericalTwoSphere,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Project a point from the ambient chart to the two-sphere intrinsic chart.

    This realization map is a special case for projecting from 3D charts to the
    two-sphere intrinsic chart, which is a common use case. The projection does
    not depend on the radius of the embedding, so this projection works in
    general.

    >>> import unxt as u
    >>> import coordinax.charts as cxc

    >>> q = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
    >>> cxc.pt_map(q, cxc.cart3d, cxc.sph2)
    {'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    """
    # Delegate to the projection map, which handles the intermediate Spherical3D
    p_s2 = cxmapi.pt_project(p, from_chart, to_chart, _twospherefrom3d, usys=usys)
    return cast("CDict", p_s2)
