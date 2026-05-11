"""Null manifold."""

__all__ = ("NoManifold", "no_manifold")


import dataclasses

from typing import Any
from typing_extensions import override

import jax.tree_util as jtu

from .atlas import NoAtlas, no_atlas
from .metric import NoMetric, no_metric
from coordinax._src.base import AbstractManifold


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
