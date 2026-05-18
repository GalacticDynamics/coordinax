"""Register dispatches for Vector."""

__all__: tuple[str, ...] = ()

from dataclasses import replace

from typing import Any

import plum

import coordinax.manifolds as cxm
from .custom_types import OptUSys
from .point import Point


@plum.dispatch
def pt_project(
    p_ambient: Point, M: cxm.HyperSphericalManifold, /, *, usys: OptUSys = None
) -> Any:
    """Project a point from an ambient space onto a manifold.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    Project a spherical 3D point onto the 2-sphere:

    >>> q = cx.Point.from_(
    ...     {"r": u.Q(1, "m"), "theta": u.Q(2, "rad"), "phi": u.Q(3, "rad")},
    ...     cx.sph3d)
    >>> cxm.pt_project(q, cxm.S2)
    Point({'theta': Q(2, 'rad'), 'phi': Q(3, 'rad')}, chart=SphericalTwoSphere(M=Sn(2)))

    """
    data = cxm.pt_project(p_ambient.data, p_ambient.chart, M, usys=usys)
    embed_map = cxm.TwoSphereIn3D(1.0)
    return replace(p_ambient, data=data, chart=embed_map.intrinsic)
