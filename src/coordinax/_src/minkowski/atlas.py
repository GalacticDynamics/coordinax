"""Minkowski spacetime manifold."""

__all__ = ("MinkowskiAtlas",)

import dataclasses

from typing import Any, ClassVar, TypeVar, final

import jax

import coordinax.charts as cxc
from coordinax._src.base import AbstractAtlas

CT = TypeVar("CT", bound=type[cxc.AbstractChart[Any, Any, Any]])


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class MinkowskiAtlas(AbstractAtlas):
    r"""Atlas of coordinate charts for Minkowski spacetime $\mathbb{R}^{1,3}$.

    An **atlas** $\mathcal{A}$ is the collection of compatible charts that
    together cover a smooth manifold. For Minkowski spacetime
    $\mathbb{R}^{1,3}$, the atlas $\mathcal{A}$ admits a chart $C$ if and
    only if:

    1. The chart dimensionality is 4.
    2. The chart class is {class}`~coordinax.charts.MinkowskiCT` (or a
       subclass explicitly registered via {meth}`register`).

    **Built-in charts:**

    - {class}`~coordinax.charts.MinkowskiCT` — canonical $(ct, x, y, z)$
      chart.

    Parameters
    ----------
    ndim : int
        Intrinsic dimension of the manifold. Always 4 for Minkowski spacetime.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> atlas = cxm.MinkowskiAtlas()
    >>> atlas.ndim
    4

    >>> cxc.minkowskict in atlas
    True

    >>> cxc.cart3d in atlas
    False

    >>> atlas.default_chart()
    MinkowskiCT(M=MinkowskiManifold(ndim=4))

    """

    ndim: int = 4
    """Dimension of Minkowski spacetime (always 4)."""

    _ELIGIBLE_CHARTS: ClassVar[set[type[cxc.AbstractChart[Any, Any, Any]]]] = set()

    def default_chart(self) -> cxc.AbstractChart[Any, Any, Any]:
        """Return the default chart (canonical ``MinkowskiCT``).

        Examples
        --------
        >>> import coordinax.manifolds as cxm
        >>> cxm.MinkowskiAtlas().default_chart()
        MinkowskiCT(M=MinkowskiManifold(ndim=4))

        """
        return cxc.minkowskict

    def has_chart(self, chart: cxc.AbstractChart[Any, Any, Any], /) -> bool:
        """Return whether the chart belongs to this atlas.

        A chart belongs when its dimensionality is 4 and its class is
        {class}`~coordinax.charts.MinkowskiCT` (or another class registered
        via {meth}`register`).

        Examples
        --------
        >>> import coordinax.manifolds as cxm
        >>> import coordinax.charts as cxc

        >>> atlas = cxm.MinkowskiAtlas()

        >>> atlas.has_chart(cxc.minkowskict)
        True

        >>> atlas.has_chart(cxc.cart3d)
        False

        """
        return chart.ndim == self.ndim and type(chart) in self._ELIGIBLE_CHARTS

    @classmethod
    def register(cls, registrant: CT, /) -> CT:
        """Register a chart class for {class}`MinkowskiAtlas` eligibility.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        >>> cxc.MinkowskiCT in cxm.MinkowskiAtlas._ELIGIBLE_CHARTS
        True

        """
        cls._ELIGIBLE_CHARTS.add(registrant)
        return registrant
