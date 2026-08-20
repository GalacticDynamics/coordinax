"""Null manifold."""

__all__ = ("NoManifold", "no_manifold")


import dataclasses

from typing import Any, override

import jax.tree_util as jtu
import plum

from .atlas import NoAtlas, no_atlas
from .metric import NoMetric, no_metric
from coordinax._src.base import AbstractChart, AbstractManifold
from coordinax._src.metric.matrix import AbstractMetricMatrix


@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class NoManifold(AbstractManifold):
    """A degenerate placeholder manifold with no charts and no geometry.

    ``NoManifold`` is a sentinel value used when a manifold object is required
    by the API but none has been specified by the user.

    - ``ndim == False`` signals "no manifold specified".
    - ``has_chart(chart)`` always returns ``False``.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> M = cxm.NoManifold()
    >>> M.ndim
    0
    >>> M.has_chart(cxc.cart2d)
    False

    """

    @override
    @property
    def atlas(self) -> NoAtlas:
        """Return the degenerate atlas on this manifold."""
        return no_atlas

    @override
    @property
    def metric(self) -> NoMetric:
        """Return the degenerate metric on this manifold."""
        return no_metric

    def has_chart(self, chart: Any, /) -> bool:
        """Return whether ``chart`` belongs to this manifold atlas."""
        return hasattr(chart, "M") and isinstance(chart.M, NoManifold)


no_manifold = NoManifold()
"""Canonical instance of `coordinax.manifolds.NoManifold`."""


@plum.dispatch
def metric_matrix(
    M: NoManifold, point: dict, chart: AbstractChart, /
) -> AbstractMetricMatrix:
    """Refuse, naming the missing manifold as the reason.

    The generic fallback advises registering a dispatch rule -- wrong for both
    populations that reach here: a chart that left `M` unset, and one that
    declares `no_manifold` by design.
    """
    del M, point
    msg = (
        f"{type(chart).__name__!r} has no manifold (`M` is `no_manifold`), so "
        "there is no metric to compute. Pair the chart with the manifold whose "
        "geometry you mean. Some charts declare no manifold by design -- a "
        "phase-space chart carries a symplectic form rather than a metric -- "
        "and for those there is no metric to ask for."
    )
    raise NotImplementedError(msg)
