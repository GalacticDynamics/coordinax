r"""Dispatch implementations for the interval / causality API.

`norm` refuses indefinite metrics, because $\sqrt{v^\top G v}$ has no real value
when the quadratic form can go negative.  The quantity that *is* well defined
there is the form itself, unrooted:

$$ \Delta s^2 = \Delta x^\top G\, \Delta x, $$

which is what this module exposes as `interval`.  For a Lorentzian metric its
**sign** is the causal character of the pair, and its magnitude gives proper
time (timelike) or proper distance (spacelike).

This is the quadratic form of the *coordinate difference*, with the metric
evaluated **at the first point** ``a`` -- not a geodesic quantity, and not the
square of `geodesic_distance`.  The two coincide only where the metric is
constant along the path: on a flat manifold, including Minkowski, the case this
module exists for.  On a curved manifold this is a first-order estimate, and it
is asymmetric in ``a`` and ``b``; `geodesic_distance` is the symmetric,
chart-invariant length, and is what to reach for when a distance is wanted.
"""

__all__: tuple[str, ...] = ()

from typing import Any

import plum

import unxt as u

import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from .quadratic_form import quadratic_form
from coordinax._src.base import (
    AbstractChart,
    AbstractMetricField,
    check_metric_is_charts,
)
from coordinax._src.custom_types import OptUSys
from coordinaxs.api.custom_types import CDict

# ===================================================================
# interval


@plum.dispatch
def interval(
    chart: AbstractChart, a: CDict, b: CDict, /, *, usys: OptUSys = None
) -> Any:
    r"""Signed squared interval between two points, in the chart's metric.

    Unlike `geodesic_distance`, this is defined for *every* metric, because it never
    takes a square root.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    For a Riemannian metric it is the squared geodesic_distance:

    >>> a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.interval(cxc.cart3d, a, b).round(2)
    Q(25., 'm2')

    For Minkowski it is negative for a timelike pair -- the case that used to
    make `geodesic_distance` return ``nan``:

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> ev = {"ct": u.Q(5.0, "m"), "x": u.Q(1.0, "m"),
    ...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.interval(cxc.minkowskict, o, ev).round(2)
    Q(-24., 'm2')

    """
    return cxmapi.interval(chart.M.metric, chart, a, b, usys=usys)


@plum.dispatch
def interval(
    metric: AbstractMetricField,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    r"""Signed squared interval with respect to an explicit metric.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> ev = {"ct": u.Q(1.0, "m"), "x": u.Q(5.0, "m"),
    ...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.interval(cxm.MinkowskiMetric(), cxc.minkowskict, o, ev).round(2)
    Q(24., 'm2')

    >>> try:
    ...     cxm.interval(cxm.FlatMetric(4), cxc.minkowskict, o, ev)
    ... except ValueError as e:
    ...     print(str(e)[:56])
    interval(): metric-level dispatch needs the chart's own

    """
    check_metric_is_charts(metric, chart, "interval")

    chart.check_data(a, keys=True, values=False)
    chart.check_data(b, keys=True, values=False)

    diff = {k: b[k] - a[k] for k in chart.components}

    # The same contraction `norm` takes the square root of, evaluated at `a`.
    # Sharing it is what gives `interval` the unit handling it would otherwise
    # have to restate -- and makes `geodesic_distance**2 == interval` hold by
    # construction rather than by the test that asserts it.
    return quadratic_form(diff, chart, at=a, usys=usys, fname="interval")


@plum.dispatch
def interval(
    chart: AbstractChart,
    a: u.AbstractQuantity,
    b: u.AbstractQuantity,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Signed squared interval for packed `unxt.Quantity` vectors.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = u.Q([0.0, 0.0, 0.0, 0.0], "m")
    >>> b = u.Q([5.0, 1.0, 0.0, 0.0], "m")
    >>> cxm.interval(cxc.minkowskict, a, b).round(2)
    Q(-24., 'm2')

    """
    return cxmapi.interval(
        chart, cxcapi.cdict(a, chart), cxcapi.cdict(b, chart), usys=usys
    )
