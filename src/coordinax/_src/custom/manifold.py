"""Customs manifolds."""

__all__ = ("CustomManifold",)


import dataclasses

from typing import final

import jax

from coordinax._src.base import AbstractAtlas, AbstractManifold, AbstractMetricField


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
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
    >>> M = cxm.CustomManifold(atlas=atlas, metric=cxm.FlatMetric(2))
    >>> M.ndim
    2
    >>> M.default_chart()
    Cart2D(M=Rn(2))
    >>> M.has_chart(cxc.polar2d)
    True

    """

    atlas: AbstractAtlas
    """Atlas defining chart compatibility for this manifold."""

    metric: AbstractMetricField
    """Riemannian metric for this manifold, used for norm and distance computations."""
