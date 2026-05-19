"""Minkowski spacetime charts."""

__all__ = ("MinkowskiCT", "minkowskict")

import dataclasses

from typing import Final, Literal as L, final  # noqa: N817
from typing_extensions import override

import jax.tree_util as jtu
import plum

from .atlas import MinkowskiAtlas
from .manifold import MinkowskiManifold
from coordinax._src.base import (
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
)
from coordinax._src.charts.d4 import Abstract4D
from coordinax._src.custom_types import Len

MinkowskiCTKeys = tuple[L["ct"], L["x"], L["y"], L["z"]]
MinkowskiCTDims = tuple[Len, Len, Len, Len]

_WRONG_M_MSG = "MinkowskiCT chart must belong to a MinkowskiManifold, got {typename}"


@MinkowskiAtlas.register
@jtu.register_static
@final
@chart_dataclass_decorator
class MinkowskiCT(
    Abstract4D,
    AbstractFixedComponentsChart[MinkowskiManifold, MinkowskiCTKeys, MinkowskiCTDims],
):
    r"""4D Minkowski spacetime chart $(ct, x, y, z)$.

    A single flat chart with the conventional 3+1 split: one time-like
    coordinate $ct$ (speed-of-light times coordinate time, carrying units of
    length) and three spatial coordinates $x$, $y$, $z$.

    All four components carry dimension ``"length"``, so the metric
    $\eta = \operatorname{diag}(-1, 1, 1, 1)$ is dimensionless.

    Parameters
    ----------
    M : AbstractManifold
        The manifold this chart belongs to.  Defaults to a fresh
        :class:`~coordinax.manifolds.MinkowskiManifold` instance.

    Examples
    --------
    >>> import coordinax.charts as cxc

    >>> cxc.MinkowskiCT()
    MinkowskiCT(M=MinkowskiManifold(ndim=4))

    >>> cxc.minkowskict.components
    ('ct', 'x', 'y', 'z')

    >>> cxc.minkowskict.coord_dimensions
    ('length', 'length', 'length', 'length')

    >>> cxc.minkowskict.ndim
    4

    """

    _: dataclasses.KW_ONLY
    M: MinkowskiManifold = MinkowskiManifold()

    def __post_init__(self) -> None:
        """Validate that M is compatible with this chart."""
        if not isinstance(self.M, MinkowskiManifold):
            raise TypeError(_WRONG_M_MSG.format(typename=type(self.M).__name__))

    @override
    @property
    def cartesian(self) -> "MinkowskiCT":
        """Return self — MinkowskiCT is already the Cartesian chart.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> cxc.minkowskict.cartesian is cxc.minkowskict
        True

        """
        return self


minkowskict: Final = MinkowskiCT()
"""Default Minkowski spacetime chart $(ct, x, y, z)$.

>>> import coordinax.charts as cxc
>>> cxc.minkowskict
MinkowskiCT()

>>> cxc.minkowskict.cartesian is cxc.minkowskict
True

"""


# ===================================================================
# guess_manifold dispatch for MinkowskiCT


@plum.dispatch
def guess_manifold(_: type[MinkowskiCT], /) -> MinkowskiManifold:
    """Infer manifold from a MinkowskiCT chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.MinkowskiCT)
    MinkowskiManifold(ndim=4)

    """
    return MinkowskiManifold()
