"""Dispatch implementations for :func:`coordinaxs.api.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()


import plum

import unxt as u
import unxts.linalg as ul

import coordinaxs.api.manifolds as cxmapi
from ._utils import as_quantity_matrix
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.custom_types import OptUSys
from coordinax._src.embedded.manifold import EmbeddedManifold
from coordinax._src.embedded.metric import PullbackMetric
from coordinax._src.metric.matrix import DiagonalMetric
from coordinaxs.api.custom_types import CDict

DMLS = u.unit("")


@plum.dispatch
def scale_factors(chart: AbstractChart, /, *, at: CDict, usys: OptUSys = None) -> ul.QM:
    """Manifold-level dispatch: delegate to the attached metric.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {
    ...     "r": u.Q(jnp.array(2.0), "km"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(jnp.array(0.0), "rad"),
    ... }
    >>> cxm.scale_factors(cxc.sph3d, at=at)
    QM([1., 4., 4.], '(, km2 / rad2, km2 / rad2)')

    """
    return cxmapi.scale_factors(chart.M.metric, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def scale_factors(
    metric: AbstractMetricField,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> ul.QM:
    """Return the diagonal entries of the metric at ``at`` in ``chart``.

    Uses the ``metric_matrix`` dispatch API to compute the metric, then
    extracts the diagonal entries.

    The chart must be orthogonal. Per-axis factors describe a metric only
    where the off-diagonal terms vanish, so a chart whose metric is dense is
    refused rather than answered with a diagonal that does not reproduce it.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.RoundMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QM([1., 1.], '(, )')

    `LonCosLatSpherical3D` is non-orthogonal -- its ``g_01 = lon_coslat *
    tan(lat)`` -- and so has no scale factors:

    >>> import unxt as u
    >>> at = {"lon_coslat": u.Angle(0.3, "rad"), "lat": u.Angle(0.6, "rad"),
    ...       "distance": u.Q(2.0, "m")}
    >>> try: cxm.scale_factors(cxc.loncoslat_sph3d, at=at)
    ... except NotImplementedError as e: print(e)
    scale_factors is a diagonal (orthogonal-frame) concept and the metric of
    LonCosLatSpherical3D... is not diagonal, so no set of per-axis factors
    reproduces it. Use coordinax.manifolds.metric_matrix.

    """
    mm = cxmapi.metric_matrix(chart.M, at, chart)
    if not isinstance(mm, DiagonalMetric):
        # Not a storage detail: a `(manifold, chart)` pair reports a diagonal
        # metric exactly when the library declares that chart orthogonal, so
        # this is the orthogonality test. See `metric_representation`.
        msg = (
            "scale_factors is a diagonal (orthogonal-frame) concept and the "
            f"metric of {type(chart).__name__} on {chart.M} is not diagonal, "
            "so no set of per-axis factors reproduces it. Use "
            "coordinax.manifolds.metric_matrix."
        )
        raise NotImplementedError(msg)

    diag = mm.diagonal
    if isinstance(diag, ul.QM):
        return diag
    units = ul.UnitsMatrix.full(diag.shape[-1], DMLS)
    return ul.QM(diag, unit=units)


@plum.dispatch
def scale_factors(
    metric: PullbackMetric, chart: AbstractChart, /, *, at: CDict, usys: OptUSys = None
) -> ul.QM:
    """Return scale factors for a pullback (induced) metric.

    The scale factors of an induced metric are the diagonal of that metric, so
    this delegates to the ``EmbeddedManifold`` ``metric_matrix`` rule rather
    than re-deriving the Jacobian. That keeps one implementation of the
    pullback, and inherits its handling of a non-intrinsic ``chart``, of a
    non-Euclidean ambient metric (the ambient Gram carries the signature), and
    of batched points.

    Points are interpreted in the passed *chart*, which need not be the
    embedding's own intrinsic chart.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
    ... )
    >>> at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> cxm.scale_factors(M.metric, cxc.sph2, at=at)
    QM([4., 4.], '(m2 / rad2, m2 / rad2)')

    A chart other than the embedding's intrinsic one:

    >>> at = {"lon": u.Angle(0.0, "rad"), "lat": u.Angle(0.0, "rad")}
    >>> cxm.scale_factors(M.metric, cxc.lonlat_sph2, at=at)
    QM([4., 4.], '(m2 / rad2, m2 / rad2)')

    The delegate reads the ambient Gram off the ambient *manifold*, so an
    ``ambient_metric`` that disagrees with it cannot be honoured. That is
    refused rather than answered with the diagonal of a different metric:

    >>> pb = cxm.PullbackMetric(cxm.TwoSphereIn3D(radius=1.0), cxm.RoundMetric(3))
    >>> try: cxm.scale_factors(pb, cxc.sph2, at=at)
    ... except NotImplementedError as e: print(e)
    the pullback of RoundMetric(ndim=3) cannot be evaluated: the ambient
    manifold Rn(3) of chart Spherical3D... carries FlatMetric(ndim=3)

    """
    del usys
    embed_map = metric.embed_map
    ambient = embed_map.ambient.M
    # `metric_matrix` below evaluates the ambient Gram on `ambient`, i.e. with
    # `ambient.metric` — the only ambient metric this route can apply. An
    # `EmbeddedManifold`'s own `metric` always agrees (it is built from
    # `ambient.metric`); a hand-built `PullbackMetric` need not.
    if metric.ambient_metric != ambient.metric:
        msg = (
            f"the pullback of {metric.ambient_metric} cannot be evaluated: the "
            f"ambient manifold {ambient} of chart {embed_map.ambient} carries "
            f"{ambient.metric}"
        )
        raise NotImplementedError(msg)
    M = EmbeddedManifold(
        intrinsic=embed_map.intrinsic.M, ambient=ambient, embed_map=embed_map
    )
    mm = cxmapi.metric_matrix(M, at, chart)
    return as_quantity_matrix(mm.matrix).diag()  # ty: ignore[unresolved-attribute]
