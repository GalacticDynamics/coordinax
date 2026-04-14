"""Customs manifolds."""

__all__ = ("CustomManifold",)

from dataclasses import dataclass

from typing import final

import jax

from .atlas import CustomAtlas
from coordinax.manifolds._src.base import AbstractManifold


@jax.tree_util.register_static
@final
@dataclass(frozen=True, slots=True)
class CustomManifold(AbstractManifold):
    r"""Smooth manifold with a caller-defined explicit atlas.

    ``CustomManifold`` is a thin wrapper around {class}`CustomAtlas` and
    inherits all chart validation and transition wrappers from
    {class}`~coordinax.manifolds.AbstractManifold`.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> atlas = cxm.CustomAtlas(
    ...     charts=(cxc.Cart2D, cxc.Polar2D),
    ...     chart_default=cxc.cart2d,
    ... )
    >>> M = cxm.CustomManifold(atlas=atlas)
    >>> M.ndim
    2
    >>> M.default_chart
    Cart2D()
    >>> M.has_chart(cxc.polar2d)
    True

    >>> x = {"x": 1.0, "y": 1.0}
    >>> M.pt_map(x, cxc.cart2d, cxc.polar2d)
    {'r': Array(1.41421356, dtype=float64, ...),
     'theta': Array(0.78539816, dtype=float64, ...)}

    """

    atlas: CustomAtlas
    """Atlas defining chart compatibility for this manifold."""

    @property
    def ndim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.atlas.ndim
