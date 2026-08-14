"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.HyperSphericalManifold` (the unit
$n$-sphere $S^n$) paired with intrinsic angular charts that derive from
:class:`~coordinax._src.spherical.chart.AbstractSphericalHyperSphere`.

The round metric on $S^n$ is diagonal in standard spherical charts, so all
rules return a :class:`~coordinax._src.metric.matrix.DiagonalMetric`.  The
diagonal entries are computed directly via the ``_sine_product_diagonal``
helper, avoiding a full-matrix allocation.

"""

__all__: tuple[str, ...] = ()

from typing import cast

import jax
import jax.numpy as jnp
import plum

import unxt as u
import unxts.linalg as ul
from unxt.quantity import AllowValue

import coordinaxs.api.charts as cxcapi
from .chart import (
    AbstractSphericalHyperSphere,
    LonCosLatSphericalTwoSphere,
    NonCanonicalTwoSphere,
    RelabeledTwoSphere,
    SphericalTwoSphere,
)
from .manifold import HyperSphericalManifold
from coordinax._src.metric.matrix import (
    DenseMetric,
    DiagonalMetric,
    _sine_product_diagonal,
)
from coordinax.internal import tree_cast_int_bool_to_float
from coordinaxs.api.custom_types import CDict

RAD = u.unit("rad")


@plum.dispatch
def metric_representation(
    M: HyperSphericalManifold, chart: AbstractSphericalHyperSphere, /
) -> type[DiagonalMetric]:
    """Return `DiagonalMetric` for a unit $n$-sphere in a standard angular chart.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.metric_representation(cxm.S2, cxc.sph2)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    """
    del M, chart
    return DiagonalMetric


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: AbstractSphericalHyperSphere, /
) -> DiagonalMetric:
    r"""Round metric on the unit $n$-sphere in a standard angular chart.

    Computes diagonal entries directly via ``_sine_product_diagonal``:

    $$g_{kk} = \prod_{j=0}^{k-1} \sin^2(\theta_j)$$

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    $S^2$ at the equator $\theta = \pi/2$:

    >>> M = cxm.HyperSphericalManifold(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> g = metric_matrix(M, at, cxc.sph2)
    >>> isinstance(g, DiagonalMetric)
    True
    >>> g.diagonal
    Array([1., 1.], dtype=float64)

    $S^2$ at $\theta = \pi/6$:

    >>> at = {"theta": jnp.array(jnp.pi / 6), "phi": jnp.array(0.0)}
    >>> g = metric_matrix(M, at, cxc.sph2)
    >>> round(float(g.diagonal[1]), 10)  # sin\u00b2(\u03c0/6) \u2248 0.25
    0.25

    """
    components = chart.components
    vals = [
        jnp.asarray(u.ustrip(AllowValue, u.uconvert_value(RAD, RAD, point[k])))
        for k in components
    ]
    # All angular components except the last (azimuthal) are polar angles.
    # Stack on the leading axis; `_sine_product_diagonal` moves it to the back.
    # The no-polar-angle case (S¹) still needs the batch shape, which only the
    # azimuthal component carries -- hence the empty stack rather than `[]`.
    thetas = (
        jnp.stack(vals[:-1])
        if len(vals) > 1
        # dtype from the input, not the environment default: otherwise the
        # empty stack decides the result dtype for S¹.
        else jnp.zeros((0, *vals[-1].shape), dtype=vals[-1].dtype)
    )
    # Dimensionless, but a `QuantityMatrix` all the same: the sibling rules
    # below already return one, and a generic consumer should not have to ask
    # which dispatch produced its metric. `angles -> angles`, so the unit is
    # empty -- unlike the *embedded* sphere, whose induced metric measures
    # ambient length and correctly carries `L**2/rad**2`.
    diag = _sine_product_diagonal(thetas, 1.0)
    n = len(components)
    return DiagonalMetric(ul.QM(diag, unit=ul.UnitsMatrix.full(n, "")))


@plum.dispatch
def metric_representation(
    M: HyperSphericalManifold, chart: LonCosLatSphericalTwoSphere, /
) -> type[DenseMetric]:
    """`LonCosLat` is non-orthogonal, so its round metric is genuinely dense.

    Its off-diagonal is ``g_01 = lon_coslat * tan(lat)``, nonzero away from the
    equator.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.metric_representation(cxm.S2, cxc.loncoslat_sph2)
    <class 'coordinax._src.metric.matrix.DenseMetric'>

    """
    del M, chart
    return DenseMetric


@plum.dispatch
def metric_representation(
    M: HyperSphericalManifold, chart: RelabeledTwoSphere, /
) -> type[DiagonalMetric]:
    """`LonLat` and `Math` relabel the canonical angles but stay orthogonal.

    Their round metrics are exactly diagonal -- ``diag(cos^2(lat), 1)`` and
    ``diag(sin^2(phi), 1)`` -- so `DiagonalMetric` is the accurate
    classification and gives callers the O(n) diagonal path.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.metric_representation(cxm.S2, cxc.lonlat_sph2)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    """
    del M, chart
    return DiagonalMetric


def _round_metric_values(point: CDict, chart: NonCanonicalTwoSphere) -> jnp.ndarray:
    """Dense ``(*batch, n, n)`` round-metric values via pullback.

    Shared by the `LonCosLat` (dense) and `Relabeled` (diagonal) rules below.
    """
    keys = chart.components
    n = len(keys)
    # Components trail: (*batch, n).
    x = tree_cast_int_bool_to_float(
        jnp.stack(
            [jnp.asarray(u.ustrip(AllowValue, RAD, point[k])) for k in keys], axis=-1
        )
    )

    canonical = SphericalTwoSphere()

    def to_canon(xi: jnp.ndarray) -> jnp.ndarray:
        p = {k: u.Q(xi[i], RAD) for i, k in enumerate(keys)}
        s = cast("CDict", cxcapi.pt_map(p, chart, canonical))
        return jnp.stack([u.ustrip(RAD, s["theta"]), u.ustrip(RAD, s["phi"])])

    def pullback(xi: jnp.ndarray) -> jnp.ndarray:
        """Metric at a single point ``xi`` of shape ``(n,)``."""
        jc = jax.jacfwd(to_canon)(xi)  # (2, n)
        theta = to_canon(xi)[0]
        g_can = jnp.diag(jnp.stack([jnp.ones_like(theta), jnp.sin(theta) ** 2]))
        return jc.T @ g_can @ jc

    # `jacfwd` on a batched input would differentiate every output w.r.t. every
    # batch element, so map it per point and restore the leading batch axes.
    batch = x.shape[:-1]
    return jax.vmap(pullback)(x.reshape(-1, n)).reshape(*batch, n, n)


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: LonCosLatSphericalTwoSphere, /
) -> DenseMetric:
    r"""Round metric on the non-orthogonal `LonCosLat` chart (pullback).

    The nested sine-product used by the canonical dispatch assumes the
    components are polar angles in nested order; that is false for these charts
    (LonLat, Math swap/relabel the polar and azimuthal angles; LonCosLat is
    non-orthogonal). The metric is instead pulled back from the canonical chart:
    ``g = Jc^T diag(1, sin^2 theta) Jc``, where ``Jc`` is the Jacobian of the
    coordinate map ``chart -> SphericalTwoSphere``.

    >>> import math
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    ``g = [[1, a], [a, 1 + a^2]]`` with ``a = lon_coslat * tan(lat)``; note the
    nonzero off-diagonal, and that ``det g == 1`` (the area element is
    ``dL dlat``):

    >>> at = {"lon_coslat": u.Q(0.4, "rad"), "lat": u.Q(0.7, "rad")}
    >>> g = cxm.metric_matrix(cxm.S2, at, cxc.loncoslat_sph2)
    >>> [round(float(v), 4) for v in g.matrix.value.ravel()]
    [1.0, 0.3369, 0.3369, 1.1135]
    >>> round(float(math.prod([0.4, math.tan(0.7)])), 4)
    0.3369

    Leading axes are batch; the two component axes trail:

    >>> import jax.numpy as jnp
    >>> at = {"lon_coslat": u.Q(jnp.zeros((5,)), "rad"),
    ...       "lat": u.Q(jnp.zeros((5,)), "rad")}
    >>> cxm.metric_matrix(cxm.S2, at, cxc.loncoslat_sph2).matrix.value.shape
    (5, 2, 2)

    .. warning::

        ``LonCosLat`` is **not a chart at the poles**: ``lon_coslat = lon *
        cos(lat)`` collapses every longitude onto ``0`` at ``lat = +-pi/2``, so
        the map is not injective there.  Away from the poles ``g_LL =
        cos^2(lat) * sec^2(lat)`` is exactly ``1`` and ``det g`` is exactly
        ``1``; evaluated *at* a pole that product becomes ``0 * inf`` and the
        returned matrix is degenerate (``det g == 0``) rather than the limiting
        identity.  Precision degrades within roughly ``1e-10`` rad of the pole,
        where the chart's condition number ``~ lon_coslat^2 tan^2(lat)`` exceeds
        what float64 can carry.  ``LonLat`` and ``Math`` are genuinely singular
        at their poles too, but there the degenerate metric *is* the correct
        limit.

    """
    del M
    n = len(chart.components)
    g = _round_metric_values(point, chart)
    dmls = ul.UnitsMatrix.full((n, n), "")  # angles -> angles, so g is dimensionless
    return DenseMetric(ul.QM(g, unit=dmls))


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: RelabeledTwoSphere, /
) -> DiagonalMetric:
    r"""Round metric on an orthogonal relabelled two-sphere chart.

    ``LonLat`` and ``Math`` relabel/swap the canonical angles, so the nested
    sine-product does not apply, but they stay orthogonal: the pullback is
    exactly diagonal, so only the diagonal is kept.

    - ``LonLat`` ``(lon, lat)``: ``diag(cos^2(lat), 1)``
    - ``Math`` ``(theta, phi)``: ``diag(sin^2(phi), 1)``

    >>> import math
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {"lon": u.Q(0.6, "rad"), "lat": u.Q(math.radians(40), "rad")}
    >>> g = cxm.metric_matrix(cxm.S2, at, cxc.lonlat_sph2)
    >>> [round(float(v), 4) for v in g.diagonal.value]
    [0.5868, 1.0]

    Leading axes are batch; the component axis trails:

    >>> import jax.numpy as jnp
    >>> at = {"lon": u.Q(jnp.zeros((5,)), "rad"), "lat": u.Q(jnp.zeros((5,)), "rad")}
    >>> cxm.metric_matrix(cxm.S2, at, cxc.lonlat_sph2).diagonal.value.shape
    (5, 2)

    """
    del M
    n = len(chart.components)
    g = _round_metric_values(point, chart)
    diag = jnp.diagonal(g, axis1=-2, axis2=-1)
    return DiagonalMetric(ul.QM(diag, unit=ul.UnitsMatrix.full(n, "")))
