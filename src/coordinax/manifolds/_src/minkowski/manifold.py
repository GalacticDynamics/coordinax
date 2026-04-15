"""Minkowski spacetime manifold."""

__all__ = (
    "MinkowskiManifold",
    "minkowski4d",
)

import dataclasses

from typing import final

import jax

from .atlas import MinkowskiAtlas
from .metric import MinkowskiMetric
from coordinax.manifolds._src.base import AbstractManifold


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class MinkowskiManifold(AbstractManifold):
    r"""Minkowski spacetime $\mathbb{R}^{1,3}$.

    **Minkowski spacetime** is the 4-dimensional pseudo-Riemannian manifold
    $(\mathbb{R}^{1,3}, \eta)$ equipped with the metric
    $\eta = \operatorname{diag}(-1, 1, 1, 1)$ in the canonical $(ct, x, y, z)$
    chart. It is the geometric arena of special relativity.

    **Charts.** The manifold admits all
    :class:`~coordinax.charts.SpaceTimeCT` charts, parameterised by the
    choice of spatial sub-chart:

    - ``SpaceTimeCT(cart3d)`` — Minkowski coordinates $(ct, x, y, z)$
    - ``SpaceTimeCT(sph3d)``  — adapted coordinates $(ct, r, \theta, \phi)$
    - ``SpaceTimeCT(cyl3d)``  — adapted coordinates $(ct, \rho, \phi, z)$

    **Metric.** All charts carry the pulled-back Minkowski metric
    $g = J^T \eta J$, where $J$ is the Jacobian from the given chart to the
    canonical Cartesian spacetime chart.

    **Pre-built instance.** The module exports :obj:`minkowski4d` as a
    ready-to-use instance.

    Parameters
    ----------
    ndim : int
        Intrinsic dimension. Always 4 for Minkowski spacetime.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> M = cxm.MinkowskiManifold()
    >>> M
    MinkowskiManifold(ndim=4)

    >>> M.ndim
    4

    >>> M.atlas
    MinkowskiAtlas(ndim=4)

    >>> M.default_chart
    SpaceTimeCT()

    >>> M.has_chart(cxc.spacetimect)
    True

    >>> M.has_chart(cxc.SpaceTimeCT(cxc.sph3d))
    True

    >>> M.has_chart(cxc.cart3d)
    False

    """

    ndim: int = 4
    """Intrinsic dimension of Minkowski spacetime (always 4)."""

    def __init__(self, ndim: int = 4, /) -> None:
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "atlas", MinkowskiAtlas(ndim))
        object.__setattr__(self, "metric", MinkowskiMetric())


minkowski4d = MinkowskiManifold()
r"""Minkowski spacetime $\mathbb{R}^{1,3}$, the arena for special relativity."""
