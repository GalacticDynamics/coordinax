"""Euclidean manifolds."""

__all__ = ("EuclideanMetric",)

import dataclasses

from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
from coordinax._src.manifolds.custom_types import CDict, OptUSys
from coordinax._src.manifolds.diagonal import AbstractDiagonalMetric
from coordinax.internal import QuantityMatrix, UnitsMatrix


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class EuclideanMetric(AbstractDiagonalMetric):
    r"""Euclidean (flat) Riemannian metric on $\mathbb{R}^n$.

    In Cartesian coordinates the metric is the identity matrix $g = I_n$.
    In any other chart, the metric matrix is computed via the pullback

    $$g_{ij} = \sum_k \frac{\partial x^k}{\partial q^i}
                           \frac{\partial x^k}{\partial q^j}
             = (J^T J)_{ij},$$

    where $J = \partial x / \partial q$ is the Jacobian of the chart-to-Cartesian
    transition map.

    This pullback is diagonal precisely for orthogonal coordinate charts.
    `EuclideanMetric` is treated as ``AbstractDiagonalMetric`` on that
    orthogonal chart domain; atlas chart compatibility alone does not imply
    orthogonality.

    Parameters
    ----------
    ndim : int
        Dimension of the Euclidean space.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> m = cxm.EuclideanMetric(3)
    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> m.metric_matrix(cxc.cart3d, at=at)
    QuantityMatrix([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]], '((, , ), (, , ), (, , ))')

    The signature is ``(1,) * ndim`` for a Riemannian (positive-definite) metric:

    >>> m.signature
    (1, 1, 1)
    >>> m.ndim
    3

    """

    ndim: int
    """Dimension of the Euclidean space."""

    @property
    def signature(self) -> tuple[int, ...]:
        """Signature of the metric as a tuple of 1's."""
        return tuple(1 for _ in range(self.ndim))

    def metric_matrix(
        self, chart: cxc.AbstractChart, /, *, at: CDict, usys: OptUSys = None
    ) -> QuantityMatrix:
        r"""Metric matrix in the given chart at the base point ``at``.

        For Cartesian charts, returns the identity matrix directly.
        For other charts, compute ``J^T J`` where ``J`` is the Jacobian of
        the curvilinear-to-Cartesian transition. This is diagonal exactly when
        the chart is orthogonal.

        """
        # Try to get the canonical Cartesian chart for this manifold
        try:
            cart_chart = chart.cartesian
        except cxc.NoGlobalCartesianChartError:
            # Chart has no Cartesian sibling; fall back to dimensionless identity
            n = self.ndim
            unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
            return QuantityMatrix(jnp.eye(n), unit=UnitsMatrix(unit_tup))

        if chart == cart_chart:
            # Already Cartesian: metric is the identity
            n = self.ndim
            unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
            return QuantityMatrix(jnp.eye(n), unit=UnitsMatrix(unit_tup))

        # Compute J = d(Cartesian)/d(chart) via jac_pt_map (returns QuantityMatrix)
        J = cxc.jac_pt_map(at, chart, cart_chart, usys=usys)
        JT = jnp.transpose(J, (1, 0))
        return JT @ J  # ty: ignore[invalid-return-type]
