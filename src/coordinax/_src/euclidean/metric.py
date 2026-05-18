"""Euclidean manifolds."""

__all__ = ("FlatMetric",)

import dataclasses

from typing import final

import jax

from coordinax._src.base import AbstractDiagonalMetricField


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class FlatMetric(AbstractDiagonalMetricField):
    r"""Euclidean (flat) Riemannian metric on $\mathbb{R}^n$.

    In Cartesian coordinates the metric is the identity matrix $g = I_n$.
    In any other chart, the metric matrix is computed via the pullback

    $$g_{ij} = \sum_k \frac{\partial x^k}{\partial q^i}
                           \frac{\partial x^k}{\partial q^j}
             = (J^T J)_{ij},$$

    where $J = \partial x / \partial q$ is the Jacobian of the chart-to-Cartesian
    transition map.

    This pullback is diagonal precisely for orthogonal coordinate charts.
    `FlatMetric` is treated as ``AbstractDiagonalMetricField`` on that
    orthogonal chart domain; atlas chart compatibility alone does not imply
    orthogonality.

    Parameters
    ----------
    ndim : int
        Dimension of the Euclidean space.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.api.manifolds as cxmapi
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> m = cxm.FlatMetric(3)
    >>> m.signature
    (1, 1, 1)
    >>> m.ndim
    3

    The metric matrix is obtained via the dispatch API on the associated manifold:

    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> cxmapi.metric_matrix(cxm.R3, at, cxc.cart3d).diagonal
    Array([1., 1., 1.], dtype=float64)

    """

    ndim: int
    """Dimension of the Euclidean space."""

    @property
    def signature(self) -> tuple[int, ...]:
        """Signature of the metric as a tuple of 1's."""
        return tuple(1 for _ in range(self.ndim))
