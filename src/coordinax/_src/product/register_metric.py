"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.CartesianProductManifold` paired with
:class:`~coordinax.charts.AbstractCartesianProductChart`.

The product metric is block-diagonal: each block is the factor metric
evaluated at the corresponding component slice of the point, computed
by recursively calling the standalone ``metric_matrix`` dispatch API.

"""

__all__: tuple[str, ...] = ()

from typing import Any

import jax.numpy as jnp
import plum

import unxt as u

from .chart import AbstractCartesianProductChart
from .manifold import CartesianProductManifold
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric
from coordinax.internal import QMatrix, UnitsMatrix
from coordinaxs.api.manifolds import metric_matrix, metric_representation

# =====================================================================
# Private helpers
# =====================================================================


def _mm_to_qm(mm: DenseMetric | DiagonalMetric) -> QMatrix:
    """Convert an AbstractMetricMatrix to a QMatrix."""
    if isinstance(mm, DiagonalMetric):
        dense = mm.to_dense()
        mat = dense.matrix
    else:
        mat = mm.matrix
    if isinstance(mat, QMatrix):
        return mat
    n = mat.shape[0]
    unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
    return QMatrix(mat, unit=UnitsMatrix(unit_tup))


# =====================================================================
# metric_representation
# =====================================================================


@plum.dispatch
def metric_representation(
    M: CartesianProductManifold, chart: AbstractCartesianProductChart, /
) -> type[DenseMetric]:
    """Product manifold in a product chart → :class:`DenseMetric`.

    The product metric is block-diagonal in general (not necessarily diagonal
    even if each factor metric is diagonal), so :class:`DenseMetric` is the
    conservative declaration.

    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_representation
    >>> from coordinax._src.metric.matrix import DenseMetric

    >>> M = cxm.CartesianProductManifold(
    ...     factors=(cxm.R2, cxm.R1), factor_names=("xy", "z")
    ... )
    >>> chart = M.default_chart()
    >>> metric_representation(M, chart)
    <class 'coordinax._src.metric.matrix.DenseMetric'>

    """
    del M, chart
    return DenseMetric


# =====================================================================
# metric_matrix
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: CartesianProductManifold, point: dict, chart: AbstractCartesianProductChart, /
) -> DenseMetric:
    r"""Product metric (block-diagonal) in a product chart.

    Assembles the block-diagonal matrix from factor metrics by recursively
    calling the standalone ``metric_matrix`` dispatch API.

    >>> import jax.numpy as jnp
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DenseMetric

    Two-factor Euclidean product (R² x R¹):

    >>> M = cxm.CartesianProductManifold(
    ...     factors=(cxm.R2, cxm.R1), factor_names=("xy", "z")
    ... )
    >>> chart = M.default_chart()
    >>> at = {k: jnp.array(0.0) for k in chart.components}
    >>> g = metric_matrix(M, at, chart)
    >>> isinstance(g, DenseMetric)
    True
    >>> g.ndim
    3

    """
    parts = chart.split_components(point)
    factor_blocks = [
        _mm_to_qm(metric_matrix(fm, fp, fc))
        for fm, fc, fp in zip(M.factors, chart.factors, parts, strict=True)
    ]

    n = sum(block.shape[0] for block in factor_blocks)
    dtype = jnp.result_type(*(block.value.dtype for block in factor_blocks))
    value = jnp.zeros((n, n), dtype=dtype)

    # First pass: place each factor block and record its diagonal units.
    diag_units: list[Any] = [u.unit("") for _ in range(n)]
    offset = 0
    for block in factor_blocks:
        block_n = block.shape[0]
        value = value.at[offset : offset + block_n, offset : offset + block_n].set(
            block.value
        )
        for i in range(block_n):
            diag_units[offset + i] = block.unit[i, i]
        offset += block_n

    # Off-diagonal (i, j) entries are numerically zero across factors, but their
    # units must be the geometric mean sqrt([g_ii]*[g_jj]) so that every term of
    # the vᵀGv contraction converts to the row reference unit g[i,0]*v[0]. This
    # mirrors DiagonalMetric.to_dense and keeps products of factors with
    # non-dimensionless metrics (e.g. an embedded sphere with a radius)
    # unit-consistent. It is a no-op when every factor metric is dimensionless.
    units = [
        [
            diag_units[i] if i == j else (diag_units[i] * diag_units[j]) ** 0.5
            for j in range(n)
        ]
        for i in range(n)
    ]
    # Second pass: restore genuine intra-block off-diagonals (dense factor metrics).
    offset = 0
    for block in factor_blocks:
        block_n = block.shape[0]
        for i in range(block_n):
            for j in range(block_n):
                units[offset + i][offset + j] = block.unit[i, j]
        offset += block_n

    unit_tup = tuple(tuple(row) for row in units)
    G = QMatrix(value=value, unit=UnitsMatrix(unit_tup))
    return DenseMetric(G)
