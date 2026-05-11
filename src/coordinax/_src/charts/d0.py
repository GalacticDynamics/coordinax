"""0-D representation."""

__all__ = ("Abstract0D", "Cart0D", "cart0d")

import dataclasses

from typing import Any, Final, Literal as L, final  # noqa: N817
from typing_extensions import override

import jax.tree_util as jtu

from coordinax._src.base_charts import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
    is_not_abstract_chart_subclass,
)
from coordinax._src.base_manifold import AbstractManifold
from coordinax._src.euclidean.atlas import (
    EUCLIDEAN_ATLAS_DEFAULT_CHARTS,
    EuclideanAtlas,
)
from coordinax._src.euclidean.manifold import euclidean0d


class Abstract0D(AbstractDimensionalFlag, n=0):
    """Marker flag for 0D representations.

    A 0D representation has no coordinate component.
    """

    # TODO: add a check it's 0D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)

        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)


# -----------------------------------------------

ZeroDKeys = tuple[()]
ZeroDDims = tuple[()]


@EuclideanAtlas.register
@jtu.register_static
@final
@chart_dataclass_decorator
class Cart0D(AbstractFixedComponentsChart[ZeroDKeys, ZeroDDims], Abstract0D):
    """Zero-dimensional Cartesian chart.

    This chart has no coordinate components and no coordinate dimensions.
    It is the canonical Cartesian chart for 0D representations.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.cart0d.components
    ()

    >>> cxc.cart0d.coord_dimensions
    ()

    >>> isinstance(cxc.cartesian_chart(cxc.cart0d), cxc.Cart0D)
    True

    """

    _: dataclasses.KW_ONLY
    M: AbstractManifold = euclidean0d

    @override
    @property
    def cartesian(self) -> "Cart0D":
        """Return the canonical Cartesian chart for a 0D chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Cart0D().cartesian, cxc.Cart0D)
        True
        """
        return self


cart0d: Final = Cart0D()
"""The canonical 0D Cartesian chart.

>>> import coordinax.charts as cxc
>>> cxc.cart0d.cartesian is cxc.cart0d
True

"""

EUCLIDEAN_ATLAS_DEFAULT_CHARTS[0] = cart0d
