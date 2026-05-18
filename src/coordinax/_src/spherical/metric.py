"""Two-sphere manifold."""

__all__ = ("RoundMetric",)

import dataclasses

from typing import final

import jax

from coordinax._src.base import AbstractDiagonalMetricField


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class RoundMetric(AbstractDiagonalMetricField):
    r"""Round metric on the unit $n$-sphere $S^{n-1}$ in standard spherical coordinates.

    The round metric on $S^2$ in the $(\theta, \phi)$ spherical chart is

    $$g = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}.$$

    Parameters
    ----------
    ndim : int
        Intrinsic dimension of the sphere (e.g. ``ndim=2`` for $S^2$).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.api.manifolds as cxmapi
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> m = cxm.RoundMetric(2)
    >>> m.signature
    (1, 1)
    >>> m.ndim
    2

    The metric matrix is obtained via the dispatch API on the associated manifold:

    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxmapi.metric_matrix(cxm.S2, at, cxc.sph2).diagonal
    Array([1., 1.], dtype=float64)

    """

    ndim: int
    """Intrinsic dimension of the sphere."""

    @property
    def signature(self) -> tuple[int, ...]:
        """Metric signature: ``(1,) * ndim`` — the round sphere metric is Riemannian."""
        return (1,) * self.ndim
