r"""Dispatch implementations for `coordinaxs.api.manifolds.chord_distance`.

The straight-line distance between two points *through the ambient space* they
are embedded in -- the tunnel rather than the surface path.

This is a different measurement from
`~coordinax.manifolds.geodesic_distance`, not an approximation to it. Both are
exact and symmetric; they answer different questions. On a sphere of radius
$R$ separated by a central angle $\theta$:

.. math::

    d_{\mathrm{geodesic}} = R\,\theta,
    \qquad
    d_{\mathrm{chord}} = 2R \sin(\theta/2).

They agree to first order and diverge as the points separate -- 0.927 against
0.964 at $\theta = 1$ on the unit sphere, and $2R$ against $\pi R$ at
antipodes. Which one is wanted depends on the question: a great-circle flight
path is the geodesic, a line of sight through the body is the chord.

A chord is only defined relative to an embedding, so this is implemented for
manifolds that carry one. A manifold that is its own ambient space -- anything
Euclidean -- has no distinct chord, and is directed to `geodesic_distance`
rather than being given the same number under a second name.
"""

__all__: tuple[str, ...] = ()

from typing import Any

import plum

import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from coordinax._src.base import AbstractChart, AbstractManifold
from coordinax._src.custom_types import OptUSys
from coordinax._src.embedded.chart import EmbeddedChart
from coordinax._src.embedded.manifold import EmbeddedManifold
from coordinax._src.euclidean.manifold import EuclideanManifold
from coordinax._src.spherical.chart import sph2
from coordinax._src.spherical.embed import TwoSphereIn3D
from coordinax._src.spherical.manifold import HyperSphericalManifold
from coordinaxs.api.custom_types import CDict


def _ambient_distance(
    embedded: EmbeddedChart[Any, Any],
    chart: AbstractChart,
    intrinsic_chart: AbstractChart,
    a: CDict,
    b: CDict,
    usys: OptUSys,
    /,
) -> Any:
    """Embed both points, then measure the straight line in the ambient space.

    The ambient manifold is flat, so its `geodesic_distance` *is* the straight
    line -- there is no separate implementation of the chord itself.

    No fast path here, deliberately. Profiling puts ~99.5% of an eager call in
    the ambient `geodesic_distance`, and ~0.5% in the embedding steps this
    could skip; within the former, the cost is a single `pt_map` between charts
    (~5ms eagerly, ~15us under `jit`). Hand-rolling the sphere's embedding
    formula here to dodge one `pt_map` would duplicate `TwoSphereIn3D` for a
    fraction of a percent.
    """
    ambient_chart = embedded.ambient

    def embed(p: CDict) -> CDict:
        intrinsic: CDict = cxcapi.pt_map(p, chart, intrinsic_chart, usys=usys)  # ty: ignore[invalid-assignment]
        out: CDict = cxmapi.pt_embed(intrinsic, embedded, usys=usys)  # ty: ignore[invalid-assignment]
        return out

    return cxmapi.geodesic_distance(
        ambient_chart.M, ambient_chart, embed(a), embed(b), usys=usys
    )


@plum.dispatch
def chord_distance(
    chart: AbstractChart, a: CDict, b: CDict, /, *, usys: OptUSys = None
) -> Any:
    """Return the ambient straight-line distance, on the chart's manifold.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    A quarter turn along the equator of the unit sphere: the arc is ``pi / 2``,
    the chord through the interior is ``sqrt(2)``.

    >>> a = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> b = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(jnp.pi / 2, "rad")}
    >>> round(float(cxm.chord_distance(cxc.sph2, a, b)), 6)
    1.414214
    >>> round(float(cxm.geodesic_distance(cxc.sph2, a, b).ustrip("rad")), 6)
    1.570796

    Antipodes are one diameter apart through the middle, half the great-circle
    distance around the outside:

    >>> n = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> s = {"theta": u.Angle(jnp.pi, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> round(float(cxm.chord_distance(cxc.sph2, n, s)), 6)
    2.0

    """
    return cxmapi.chord_distance(chart.M, chart, a, b, usys=usys)


@plum.dispatch
def chord_distance(
    M: HyperSphericalManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Return the chord of the unit hypersphere, through its canonical embedding.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    Any chart on the sphere gives the same answer:

    >>> a = {"lon": u.Angle(0.0, "rad"), "lat": u.Angle(0.0, "rad")}
    >>> b = {"lon": u.Angle(jnp.pi / 2, "rad"), "lat": u.Angle(0.0, "rad")}
    >>> round(float(cxm.chord_distance(cxc.lonlat_sph2, a, b)), 6)
    1.414214

    """
    if M.ndim != 2:
        msg = (
            f"chord_distance is only implemented for the two-sphere; {M} is "
            f"{M.ndim}-dimensional."
        )
        raise NotImplementedError(msg)
    unit_sphere = EmbeddedChart(TwoSphereIn3D(radius=1.0))
    return _ambient_distance(unit_sphere, chart, sph2, a, b, usys)


@plum.dispatch
def chord_distance(
    M: EmbeddedManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Return the chord through the manifold's own ambient space.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    A sphere of radius 2 m: antipodes are one diameter apart.

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
    ... )
    >>> n = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> s = {"theta": u.Angle(jnp.pi, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> cxm.chord_distance(M, cxc.sph2, n, s).round(6)
    Distance(4., 'm')

    """
    embedded = EmbeddedChart(M.embed_map)
    return _ambient_distance(embedded, chart, M.embed_map.intrinsic, a, b, usys)


@plum.dispatch
def chord_distance(
    M: EuclideanManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Refuse: flat space is its own ambient, so the chord is the geodesic.

    Returning the same number under a second name invites the reader to think
    two things were measured.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> try: cxm.chord_distance(cxc.cart3d, a, b)
    ... except NotImplementedError as e: print(e)
    chord_distance is a measurement through an ambient space, and Rn(3) is its
    own ambient -- its chord is the straight line, which is what
    `geodesic_distance` already returns.

    """
    del chart, a, b, usys
    msg = (
        "chord_distance is a measurement through an ambient space, and "
        f"{M} is its own ambient -- its chord is the straight line, which is "
        "what `geodesic_distance` already returns."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def chord_distance(
    M: AbstractManifold,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Refuse: without an embedding there is no ambient space to cut through."""
    del chart, a, b, usys
    msg = (
        f"no chord distance is implemented for {M}: a chord is measured "
        "through an ambient space, and this manifold carries no embedding. "
        "Wrap it in an `EmbeddedManifold`, or use `geodesic_distance` for the "
        "distance along the manifold."
    )
    raise NotImplementedError(msg)
