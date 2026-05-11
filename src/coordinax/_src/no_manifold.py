"""Manifold definitions and manifold inference helpers."""

__all__ = ("NoManifold", "no_manifold")


import dataclasses

from typing import Any

import jax.tree_util as jtu

from .base_topo import AbstractTopologicalManifold


@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class NoManifold(AbstractTopologicalManifold):
    """A degenerate placeholder manifold with no charts and no geometry.

    ``NoManifold`` is a sentinel value used when a manifold object is required
    by the API but none has been specified by the user.

    - ``ndim == -1`` signals "no manifold specified".
    - ``has_chart(chart)`` always returns ``False``.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> M = cxm.NoManifold()
    >>> M.ndim
    False
    >>> M.has_chart(cxc.cart2d)
    False

    """

    ndim: int = False
    """Stand-in dimension of the degenerate manifold."""

    def has_chart(self, chart: Any, /) -> bool:
        """Return whether ``chart`` belongs to this manifold atlas."""
        return hasattr(chart, "M") and isinstance(chart.M, NoManifold)


no_manifold = NoManifold()
"""Canonical instance of `coordinax.manifolds.NoManifold`."""
