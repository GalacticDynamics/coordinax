"""Register dispatches for Vector."""

__all__: tuple[str, ...] = ()

from dataclasses import replace

from typing import Any

import plum

import coordinax.charts as cxc
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
    >>> import coordinax as cx
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    Project a spherical 3D point onto the 2-sphere:

    >>> q = cx.Point.from_(
    ...     {"r": u.Q(1, "m"), "theta": u.Q(2, "rad"), "phi": u.Q(3, "rad")},
    ...     cx.sph3d)
    >>> cxm.pt_project(q, cxm.S2)
    Point(
      {'theta': Angle(2, 'rad'), 'phi': Angle(3, 'rad')},
      chart=SphericalTwoSphere(M=Sn(2))
    )

    """
    data = cxm.pt_project(p_ambient.data, p_ambient.chart, M, usys=usys)
    embed_map = cxm.TwoSphereIn3D(1.0)
    return replace(p_ambient, data=data, chart=embed_map.intrinsic)


@plum.dispatch
def metric_matrix(
    M: cxm.AbstractManifold, point: Point, chart: cxc.AbstractChart, /
) -> Any:  # type: ignore[type-arg]
    """Evaluate the metric at a `Point`, expressed in ``chart``.

    The point is converted to ``chart`` first, so a base point given in any chart
    of the manifold evaluates the metric at the same place. This is what lets
    ``at=`` be a `Point` in `~coordinax.manifolds.norm` and the other metric
    measurements, not only a component dict.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> p = cx.Point.from_([0.0, 2.0, 0.0], "m")
    >>> v = {"r": u.Q(0.0, "m/s"), "theta": u.Q(0.0, "rad/s"),
    ...      "phi": u.Q(1.0, "rad/s")}
    >>> cxm.norm(v, cxc.sph3d, at=p).round(6)
    Q(2., 'm / s')

    """
    return cxm.metric_matrix(M, point.cconvert(chart).data, chart)
