"""Two-sphere manifold."""

__all__ = ("HyperSphericalManifold", "twosphere")

from dataclasses import dataclass

from typing import final

import jax

from .atlas import HyperSphericalAtlas
from .metric import HyperSphericalMetric
from coordinax._src.manifolds.base import AbstractManifold


@jax.tree_util.register_static
@final
@dataclass(frozen=True, slots=True)
class HyperSphericalManifold(AbstractManifold):
    r"""The unit two-sphere $S^2$ as a smooth manifold.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> S2 = cxm.HyperSphericalManifold()
    >>> S2.ndim
    2

    >>> S2.has_chart(cxc.sph2)
    True

    >>> S2.default_chart
    SphericalTwoSphere()

    """

    ndim: int = 2
    """Intrinsic dimension of the manifold."""

    def __init__(self, ndim: int = 2, /) -> None:
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "atlas", HyperSphericalAtlas(self.ndim))
        object.__setattr__(self, "metric", HyperSphericalMetric(self.ndim))


twosphere = HyperSphericalManifold(2)  # Reusable instance of the two-sphere manifold
r"""The spherical manifold, e.g. $S^2$."""
