"""Dispatch implementations for `coordinaxs.api.manifolds.separation`.

The straight-line distance between two points is the manifold
`~coordinax.manifolds.norm` of their coordinate difference, evaluated in the
chart they are given in.  This is exact for a flat manifold (e.g. the Euclidean
distance in a Cartesian chart); in a curvilinear chart it is the norm of the
coordinate difference at the first point, not the geodesic distance, so bring
the points into a Cartesian chart first (as the `~coordinax.vectors.Point`
overload does) for a chart-invariant result.
"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any

import plum

import unxt as u

import coordinax.distances as cxd
import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.custom_types import CDict, OptUSys

_LENGTH = u.dimension("length")


def _as_distance(dist: Any) -> Any:
    """Wrap a length magnitude as a `Distance`; pass anything else through."""
    if hasattr(dist, "unit") and u.dimension_of(dist) == _LENGTH:
        return cxd.Distance.from_(dist)
    return dist


@plum.dispatch
def separation(
    chart: AbstractChart, a: CDict, b: CDict, /, *, usys: OptUSys = None
) -> Any:
    """Distance between two points, using the chart manifold's metric.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.separation(cxc.cart3d, a, b).round(2)
    Distance(5., 'm')

    """
    return cxmapi.separation(chart.M.metric, chart, a, b, usys=usys)


@plum.dispatch
def separation(
    metric: AbstractMetricField,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Distance between two points with respect to an explicit metric.

    The distance is the `~coordinax.manifolds.norm` of ``b - a`` (evaluated at
    ``a``); a length result is returned as a `Distance`, a dimensionless one as a
    bare array.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.FlatMetric(3)
    >>> a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.separation(metric, cxc.cart3d, a, b).round(2)
    Distance(5., 'm')

    """
    chart.check_data(a, keys=True, values=False)
    chart.check_data(b, keys=True, values=False)
    diff = {k: b[k] - a[k] for k in chart.components}
    return _as_distance(cxmapi.norm(diff, metric, chart, at=a, usys=usys))


@plum.dispatch
def separation(
    chart: AbstractChart,
    a: u.AbstractQuantity,
    b: u.AbstractQuantity,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Distance between two points given as packed `unxt.Quantity` vectors.

    Each quantity's trailing axis holds the components in ``chart.components``
    order; it is unpacked into a component dictionary and delegated to the
    ``CDict`` overload.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = u.Q([3.0, 0.0, 0.0], "m")
    >>> b = u.Q([0.0, 4.0, 0.0], "m")
    >>> cxm.separation(cxc.cart3d, a, b).round(2)
    Distance(5., 'm')

    """
    return cxmapi.separation(
        chart, cxcapi.cdict(a, chart), cxcapi.cdict(b, chart), usys=usys
    )


@plum.dispatch
def separation(
    chart: AbstractChart, a: Array, b: Array, /, *, usys: OptUSys = None
) -> Any:
    """Distance between two points given as packed (unitless) arrays.

    The trailing axis holds the components in ``chart.components`` order.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = jnp.array([3.0, 0.0, 0.0])
    >>> b = jnp.array([0.0, 4.0, 0.0])
    >>> float(cxm.separation(cxc.cart3d, a, b))
    5.0

    """
    return cxmapi.separation(
        chart, cxcapi.cdict(a, chart), cxcapi.cdict(b, chart), usys=usys
    )
