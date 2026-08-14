"""Dispatch implementations for `coordinaxs.api.manifolds.geodesic_distance`.

The length of the shortest path between two points *along the manifold*.

This is a genuine metric: symmetric, and independent of the chart the points
are handed over in. It is therefore computed per manifold, from that
manifold's geometry, rather than by measuring a coordinate difference:

- `~coordinax.manifolds.EuclideanManifold`: the straight-line Euclidean
  distance, evaluated in the manifold's Cartesian chart.
- `~coordinax.manifolds.HyperSphericalManifold` and the embedded 2-sphere: the
  great-circle distance, i.e. the central angle scaled by the radius.

A manifold with no closed-form geodesic raises `NotImplementedError` rather
than returning an approximation. Note in particular what this is *not*: the
norm of the coordinate difference, ``||b - a||_a``. That quantity is only a
first-order estimate away from a flat manifold in a Cartesian chart, and it is
asymmetric -- on the unit sphere it disagrees with itself by 10% under a swap
of the arguments, and runs ~6% low near antipodal. `~coordinax.manifolds.interval`
is the signed square of it, and stays defined that way: it exists for
Minkowski, where the metric is constant and the estimate is exact.
"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any

import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.distances as cxd
import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from coordinax._src.base import (
    AbstractChart,
    AbstractManifold,
    AbstractMetricField,
    check_metric_is_charts,
)
from coordinax._src.charts.d3 import cart3d
from coordinax._src.custom_types import OptUSys
from coordinax._src.embedded.chart import EmbeddedChart
from coordinax._src.euclidean.manifold import EuclideanManifold
from coordinax._src.minkowski.manifold import MinkowskiManifold
from coordinax._src.spherical.chart import sph2
from coordinax._src.spherical.embed import TwoSphereIn3D
from coordinax._src.spherical.manifold import HyperSphericalManifold
from coordinaxs.api.custom_types import CDict

_LENGTH = u.dimension("length")


def _as_distance(dist: Any) -> Any:
    """Wrap a length magnitude as a `Distance`; pass anything else through."""
    if hasattr(dist, "unit") and u.dimension_of(dist) == _LENGTH:
        return cxd.Distance.from_(dist)
    return dist


@plum.dispatch
def geodesic_distance(
    chart: AbstractChart, a: CDict, b: CDict, /, *, usys: OptUSys = None
) -> Any:
    """Geodesic distance between two points, on the chart's manifold.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    A 3-4-5 triangle in flat space:

    >>> a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.geodesic_distance(cxc.cart3d, a, b).round(2)
    Distance(5., 'm')

    The same two points in a curvilinear chart give the same answer, which the
    norm of their coordinate difference would not:

    >>> a_sph = cxc.pt_map(a, cxc.cart3d, cxc.sph3d)
    >>> b_sph = cxc.pt_map(b, cxc.cart3d, cxc.sph3d)
    >>> cxm.geodesic_distance(cxc.sph3d, a_sph, b_sph).round(2)
    Distance(5., 'm')

    On the unit sphere it is the great-circle distance, and symmetric:

    >>> import jax.numpy as jnp
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> q = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(1.0, "rad")}
    >>> round(float(cxm.geodesic_distance(cxc.sph2, p, q).ustrip("rad")), 6)
    1.0
    >>> round(float(cxm.geodesic_distance(cxc.sph2, q, p).ustrip("rad")), 6)
    1.0

    """
    return cxmapi.geodesic_distance(chart.M, chart, a, b, usys=usys)


@plum.dispatch
def geodesic_distance(
    M: EuclideanManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Return the straight-line distance, measured in the Cartesian chart.

    Mapping both points into the manifold's Cartesian chart first is what makes
    this chart-invariant and symmetric. Measuring ``||b - a||`` in the chart the
    caller happened to use is neither, unless that chart is already Cartesian.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = {"r": u.Q(2.0, "m"), "theta": u.Angle(1.0, "rad"),
    ...      "phi": u.Angle(0.4, "rad")}
    >>> b = {"r": u.Q(3.0, "m"), "theta": u.Angle(1.3, "rad"),
    ...      "phi": u.Angle(0.9, "rad")}
    >>> round(float(cxm.geodesic_distance(cxc.sph3d, a, b).ustrip("m")), 6)
    1.651376

    """
    del M
    cart = chart.cartesian
    ca: CDict = cxcapi.pt_map(a, chart, cart, usys=usys)  # ty: ignore[invalid-assignment]
    cb: CDict = cxcapi.pt_map(b, chart, cart, usys=usys)  # ty: ignore[invalid-assignment]
    diff = {k: cb[k] - ca[k] for k in cart.components}
    return _as_distance(cxmapi.norm(diff, cart.M.metric, cart, at=ca, usys=usys))


def _central_angle(ua: CDict, ub: CDict, keys: tuple[str, ...], /) -> Any:
    """Angle between two unit vectors, from their ambient Cartesian components.

    Uses ``atan2(|u - (u.v) v|, u.v)`` rather than ``arccos(u.v)``: ``arccos``
    loses most of its significant digits for nearby points, where ``u.v``
    approaches 1, which is the regime a distance is most often wanted in.
    """
    dot = sum(ua[k] * ub[k] for k in keys)
    perp = jnp.sqrt(sum((ua[k] - dot * ub[k]) ** 2 for k in keys))
    return jnp.atan2(perp, dot)


@plum.dispatch
def geodesic_distance(
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
    >>> cxm.geodesic_distance(cxc.cart3d, a, b).round(2)
    Distance(5., 'm')

    """
    return cxmapi.geodesic_distance(
        chart, cxcapi.cdict(a, chart), cxcapi.cdict(b, chart), usys=usys
    )


@plum.dispatch
def geodesic_distance(
    chart: AbstractChart, a: Array, b: Array, /, *, usys: OptUSys = None
) -> Any:
    """Distance between two points given as packed (unitless) arrays.

    The trailing axis holds the components in ``chart.components`` order.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = jnp.array([3.0, 0.0, 0.0])
    >>> b = jnp.array([0.0, 4.0, 0.0])
    >>> float(cxm.geodesic_distance(cxc.cart3d, a, b))
    5.0

    """
    return cxmapi.geodesic_distance(
        chart, cxcapi.cdict(a, chart), cxcapi.cdict(b, chart), usys=usys
    )


@plum.dispatch
def geodesic_distance(
    M: HyperSphericalManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Return the great-circle distance on the unit hypersphere.

    The points are embedded into the ambient Cartesian space as unit vectors
    and the central angle between them is taken, which for a unit sphere *is*
    the arc length. Any chart on the sphere works: the points are routed
    through the embedding's own intrinsic chart first, so the answer does not
    depend on which one the caller used.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    A quarter turn along the equator is `pi / 2`:

    >>> a = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> b = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(jnp.pi / 2, "rad")}
    >>> round(float(cxm.geodesic_distance(cxc.sph2, a, b).ustrip("rad")), 6)
    1.570796

    Antipodes are `pi` apart -- the coordinate-difference norm cannot reach
    this, since it has no way to know the sphere closes up:

    >>> n = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sth = {"theta": u.Angle(jnp.pi, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> round(float(cxm.geodesic_distance(cxc.sph2, n, sth).ustrip("rad")), 6)
    3.141593

    """
    if M.ndim != 2:
        msg = (
            f"geodesic_distance is only implemented for the two-sphere; {M} is "
            f"{M.ndim}-dimensional. Embed it and use the ambient distance, or "
            "add a rule for this manifold."
        )
        raise NotImplementedError(msg)

    unit_sphere = EmbeddedChart(TwoSphereIn3D(radius=u.Q(1.0, "")))
    keys = ("x", "y", "z")

    def to_unit_vector(p: CDict) -> CDict:
        intrinsic: CDict = cxcapi.pt_map(p, chart, sph2, usys=usys)  # ty: ignore[invalid-assignment]
        ambient: CDict = cxmapi.pt_embed(intrinsic, unit_sphere, usys=usys)  # ty: ignore[invalid-assignment]
        out: CDict = cxcapi.pt_map(ambient, unit_sphere.ambient, cart3d, usys=usys)  # ty: ignore[invalid-assignment]
        return out

    # Arc length is radius * angle, and the radius here is 1, so the arc
    # length *is* the central angle -- reported in radians rather than as a
    # bare number, because that is what it is.
    return _central_angle(to_unit_vector(a), to_unit_vector(b), keys)


@plum.dispatch
def geodesic_distance(
    M: AbstractManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Refuse: no closed-form geodesic is known for this manifold.

    Refused rather than approximated: the norm of the coordinate difference,
    which this used to return, is asymmetric on a curved manifold and so is not
    a distance at all.
    """
    del chart, a, b, usys
    msg = (
        f"no geodesic distance is implemented for {M}. Implementations exist "
        "for Euclidean manifolds and the two-sphere; for anything else, "
        "integrate the geodesic yourself, or use `interval` for the "
        "first-order signed square of the coordinate difference."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def geodesic_distance(
    M: MinkowskiManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Refuse: a pseudo-Riemannian manifold has no Riemannian distance.

    The Minkowski metric is indefinite, so ``g(v, v)`` is negative for a
    timelike pair and its square root is not a length. Returning ``nan`` there
    while returning a plausible number for a spacelike pair -- which is what
    taking the root unguarded does -- hides the failure in exactly the half of
    spacetime a reader is least likely to probe.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> origin = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> event = {"ct": u.Q(5.0, "m"), "x": u.Q(1.0, "m"),
    ...          "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> try: cxm.geodesic_distance(cxc.minkowskict, origin, event)
    ... except NotImplementedError as e: print(e)
    geodesic_distance() requires a Riemannian (positive-definite) metric;
    MinkowskiManifold(ndim=4) is pseudo-Riemannian, whose indefinite metric
    admits no distance. Use `interval` for the signed square, `proper_time`
    for a timelike pair, or `proper_distance` for a spacelike one.

    """
    del chart, a, b, usys
    msg = (
        "geodesic_distance() requires a Riemannian (positive-definite) metric; "
        f"{M} is pseudo-Riemannian, whose indefinite metric admits no "
        "distance. Use `interval` for the signed square, `proper_time` for a "
        "timelike pair, or `proper_distance` for a spacelike one."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def geodesic_distance(
    metric: AbstractMetricField,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Geodesic distance with the metric stated explicitly.

    The geodesic is a property of the manifold, so this checks that ``metric``
    is the one ``chart`` carries and then defers to the manifold rule. It
    exists so a caller can be explicit, and so a mismatched metric is refused
    rather than silently replaced by the chart's.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.geodesic_distance(cxm.FlatMetric(3), cxc.cart3d, a, b).round(2)
    Distance(5., 'm')

    """
    check_metric_is_charts(metric, chart, "geodesic_distance")
    return cxmapi.geodesic_distance(chart.M, chart, a, b, usys=usys)
