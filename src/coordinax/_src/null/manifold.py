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
    """Refuse, naming the manifold as the reason.

    The generic fallback tells the caller to register a rule. That is the wrong
    advice here: `NoManifold` is the sentinel for "no manifold specified", and a
    chart that declares one -- `~coordinax.charts.PoincarePolar6D`, whose phase
    space carries a symplectic form rather than a metric -- has no metric to
    register in the first place.
    """
    del M, point
    msg = (
        f"{type(chart).__name__!r} has no manifold (`M` is `no_manifold`), so it "
        "has no metric to compute. A chart on phase space carries a symplectic "
        "form rather than a metric; if this chart should have one, pair it with "
        "the manifold whose geometry you mean."
    )
    raise NotImplementedError(msg)
