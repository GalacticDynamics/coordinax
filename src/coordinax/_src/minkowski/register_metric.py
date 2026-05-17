r"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.MinkowskiManifold` (Minkowski spacetime
$\mathbb{R}^{1,3}$) paired with charts in the Minkowski atlas.

* For the canonical :class:`~coordinax.charts.MinkowskiCT` chart
  ``(ct, x, y, z)`` the metric is $\eta = \operatorname{diag}(-1, 1, 1, 1)$,
  returned as a :class:`~coordinax._src.metric.matrix.DiagonalMetric`.
* For all other registered charts the rule computes the pullback
  $g = J^T \eta J$ via :func:`~coordinax.charts.jac_pt_map` directly
  and wraps the result in a :class:`~coordinax._src.metric.matrix.DenseMetric`.

"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import plum

import unxt as u

import coordinax.charts as cxc
from .charts import MinkowskiCT
from .manifold import MinkowskiManifold
from coordinax._src.base import AbstractChart  # type: ignore[type-arg]
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric
from coordinax.api.manifolds import metric_matrix, metric_representation
from coordinax.internal import QMatrix, UnitsMatrix

# =====================================================================
# metric_representation
# =====================================================================


@plum.dispatch
def metric_representation(
    M: MinkowskiManifold, chart: MinkowskiCT, /
) -> type[DiagonalMetric]:
    """Minkowski manifold in the canonical CT chart → :class:`DiagonalMetric`.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.api.manifolds import metric_representation
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> M = cxm.MinkowskiManifold()
    >>> metric_representation(M, cxc.minkowskict)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    """
    del M, chart
    return DiagonalMetric


@plum.dispatch
def metric_representation(
    M: MinkowskiManifold, chart: AbstractChart, /
) -> type[DenseMetric]:
    """Minkowski manifold in a general chart → :class:`DenseMetric`.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.api.manifolds import metric_representation
    >>> from coordinax._src.metric.matrix import DenseMetric

    >>> M = cxm.MinkowskiManifold()
    >>> metric_representation(M, cxc.minkowskict)  # doctest: +SKIP
    <class 'coordinax._src.metric.matrix.DenseMetric'>

    """
    del M, chart
    return DenseMetric


# =====================================================================
# metric_matrix — canonical Minkowski CT chart
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: MinkowskiManifold, point: dict, chart: MinkowskiCT, /
) -> DiagonalMetric:
    r"""Minkowski metric $\eta = \operatorname{diag}(-1, 1, 1, 1)$ in CT chart.

    All four components ``(ct, x, y, z)`` carry dimension ``"length"``, so the
    entries are dimensionless.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> M = cxm.MinkowskiManifold()
    >>> at = {"ct": jnp.array(0.0), "x": jnp.array(1.0),
    ...       "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> g = metric_matrix(M, at, cxc.minkowskict)
    >>> isinstance(g, DiagonalMetric)
    True
    >>> g.diagonal
    Array([-1.,  1.,  1.,  1.], dtype=float64)

    """
    del M, point, chart
    return DiagonalMetric(jnp.array([-1.0, 1.0, 1.0, 1.0]))


# =====================================================================
# metric_matrix — general fallback
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: MinkowskiManifold, point: dict, chart: AbstractChart, /
) -> DenseMetric:
    r"""Minkowski metric in a general chart via Jacobian pullback $g = J^T \eta J$.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    Canonical chart uses the specific dispatch above, not this fallback:

    >>> M = cxm.MinkowskiManifold()
    >>> at = {"ct": jnp.array(0.0), "x": jnp.array(1.0),
    ...       "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> g = metric_matrix(M, at, cxc.minkowskict)
    >>> isinstance(g, DiagonalMetric)
    True

    """
    n = 4
    unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
    cart_chart = chart.cartesian
    J = cxc.jac_pt_map(point, chart, cart_chart, usys=None)
    JT = J.T
    eta = QMatrix(
        jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0])),
        unit=UnitsMatrix(unit_tup),
    )
    return DenseMetric(JT @ eta @ J)
