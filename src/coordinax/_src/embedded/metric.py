"""Representations for embedded manifolds."""

__all__ = ("PullbackMetric",)

import dataclasses

from typing import final

import jax

import unxt as u

from .embedmap import AbstractEmbeddingMap
from coordinax._src.base import AbstractMetricField

DMLS = u.unit("")


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class PullbackMetric(AbstractMetricField):
    r"""Pullback metric induced by an embedding map.

    Given an embedding $\iota : N \hookrightarrow M$, the metric $g_N$ on the
    submanifold is the pullback of the ambient metric $g_M$:

    $$g_N = \iota^* g_M, \quad \text{or component-wise}\quad
      (g_N)_{ij} = (J^T G J)_{ij},$$

    where $J = \partial \iota / \partial q$ is the Jacobian of the embedding
    map and $G = g_M$ is the ambient metric evaluated at $\iota(p)$.

    Parameters
    ----------
    embed_map : AbstractEmbeddingMap
        The embedding map from the submanifold into the ambient space.
    ambient_metric : AbstractMetricField
        The Riemannian metric on the ambient manifold.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.api.manifolds as cxmapi
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> embed_map = cxm.TwoSphereIn3D(radius=u.Q(1.0, "km"))
    >>> ambient_metric = cxm.FlatMetric(3)
    >>> M = cxm.PullbackMetric(embed_map, ambient_metric)
    >>> M.signature
    (1, 1)
    >>> M.ndim
    2

    The metric matrix is obtained via the dispatch API on an
    :class:`~coordinax.manifolds.EmbeddedManifold`:

    >>> M_emb = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
    ... )
    >>> at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
    >>> g = cxmapi.metric_matrix(M_emb, at, cxc.sph2)
    >>> g.matrix.value
    Array([[1., 0.],
           [0., 1.]], dtype=float64)
    >>> g.matrix.unit[0, 0]
    Unit("km2 / rad2")

    """

    embed_map: AbstractEmbeddingMap
    ambient_metric: AbstractMetricField

    @property
    def ndim(self) -> int:
        """Dimension of the submanifold."""
        return self.embed_map.intrinsic.ndim

    @property
    def signature(self) -> tuple[int, ...]:
        """Metric signature ``(1,) * m`` where ``m`` is the intrinsic dimension.

        Embedding into a Riemannian ambient manifold always produces a
        Riemannian induced metric (``J^T g_M J`` is positive-definite when
        ``J`` has full column rank).  An indefinite ambient carries no such
        guarantee — a curve in Minkowski space is timelike or spacelike
        depending on where it goes, not on its dimension — so there is no
        answer to give from the embedding alone.
        """
        if any(s < 0 for s in self.ambient_metric.signature):
            msg = (
                "the signature of a pullback from the indefinite ambient metric "
                f"{self.ambient_metric} depends on the embedding, not only on its "
                "dimension; evaluate `metric_matrix` at a point instead"
            )
            raise NotImplementedError(msg)
        return (1,) * self.ndim
