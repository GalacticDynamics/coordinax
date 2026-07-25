"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.CartesianProductManifold` paired with
:class:`~coordinax.charts.AbstractCartesianProductChart`.

The product metric is block-diagonal: each block is the factor metric
evaluated at the corresponding component slice of the point, computed
by recursively calling the standalone ``metric_matrix`` dispatch API.

"""

__all__: tuple[str, ...] = ()

from itertools import combinations, product

from typing import Any

import jax.numpy as jnp
import plum
import unxts.linalg as ul

import unxt as u

from .chart import AbstractCartesianProductChart
from .manifold import CartesianProductManifold
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric
from coordinaxs.api.manifolds import metric_matrix, metric_representation

# =====================================================================
# Private helpers
# =====================================================================


def _mm_to_qm(mm: DenseMetric | DiagonalMetric) -> ul.QuantityMatrix:
    """Convert an AbstractMetricMatrix to a QuantityMatrix."""
    if isinstance(mm, DiagonalMetric):
        dense = mm.to_dense()
        mat = dense.matrix
    else:
        mat = mm.matrix
    if isinstance(mat, ul.QuantityMatrix):
        return mat
    n = mat.shape[0]
    unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
    return ul.QuantityMatrix(mat, unit=ul.UnitsMatrix(unit_tup))


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

    # Place each factor's numeric block and its (intra-block) units; record the
    # index range of each block for the cross-factor pass below.
    units: list[list[Any]] = [[u.unit("") for _ in range(n)] for _ in range(n)]
    block_ranges: list[range] = []
    offset = 0
    for block in factor_blocks:
        block_n = block.shape[0]
        value = value.at[offset : offset + block_n, offset : offset + block_n].set(
            block.value
        )
        for i in range(block_n):
            for j in range(block_n):
                units[offset + i][offset + j] = block.unit[i, j]
        block_ranges.append(range(offset, offset + block_n))
        offset += block_n

    # Cross-factor (i, j) entries are numerically zero, but their units must be
    # the geometric mean sqrt([g_ii]*[g_jj]) so that every term of the vᵀGv
    # contraction converts to the row reference unit g[i,0]*v[0]. This mirrors
    # DiagonalMetric.to_dense and keeps products of factors with non-
    # dimensionless metrics (e.g. an embedded sphere with a radius) consistent.
    # Only cross-block entries are filled (intra-block units are kept), so it
    # is a no-op when every factor metric is dimensionless.
    #
    # The mean factorises as sqrt([g_ii]*[g_jj]) = sqrt([g_ii]) * sqrt([g_jj]),
    # so the per-index sqrt is taken once (n roots, not one per pair).
    sqrt_diag = [units[i][i] ** 0.5 for i in range(n)]
    for ra, rb in combinations(block_ranges, 2):
        for i, j in product(ra, rb):
            units[i][j] = units[j][i] = sqrt_diag[i] * sqrt_diag[j]

    unit_tup = tuple(tuple(row) for row in units)
    G = ul.QuantityMatrix(value=value, unit=ul.UnitsMatrix(unit_tup))
    return DenseMetric(G)
