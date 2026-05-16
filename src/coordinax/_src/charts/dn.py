"""N-Dimensional charts."""

__all__ = ("AbstractND", "CartND", "cartnd")


import dataclasses

from typing import Any, Final, Literal as L, final  # noqa: N817
from typing_extensions import override

import jax.tree_util as jtu

from coordinax._src.base import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    AbstractManifold,
    chart_dataclass_decorator,
    is_not_abstract_chart_subclass,
)
from coordinax._src.custom_types import Len
from coordinax._src.euclidean.atlas import EuclideanAtlas
from coordinax._src.null import no_manifold


class AbstractND(AbstractDimensionalFlag, n="N"):
    """Marker flag for N-D representations.

    An N-D representation has an arbitrary number of coordinate components.
    Examples include Cartesian representations in arbitrary dimensions.
    """

    _: dataclasses.KW_ONLY
    M: AbstractManifold = no_manifold

    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)
        # n is already fixed to "N"
        if n is not None:
            msg = f"{cls.__name__} does not support fixed n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)


# -----------------------------------------------
# Cartesian

CartNDKeys = tuple[L["q"]]
CartNDDims = tuple[Len]


@EuclideanAtlas.register
@jtu.register_static
@final
@chart_dataclass_decorator
class CartND(AbstractFixedComponentsChart[CartNDKeys, CartNDDims], AbstractND):
    r"""N-dimensional Cartesian chart.

    Components are ordered as ``("q",)`` with dimension ``("length",)``,
    where ``q`` stores the N Cartesian components as a single length-valued
    array.

    This chart is the canonical Cartesian chart for arbitrary dimension.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.cartnd.components
    ('q',)

    >>> cxc.cartnd.coord_dimensions
    ('length',)

    >>> isinstance(cxc.cartesian_chart(cxc.cartnd), cxc.CartND)
    True

    """

    _: dataclasses.KW_ONLY
    M: AbstractManifold = RN

    @override
    @property
    def cartesian(self) -> "CartND":
        """Return the canonical Cartesian chart for an N-D chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.CartND().cartesian, cxc.CartND)
        True

        """
        return self


cartnd: Final = CartND()
"""The canonical N-D Cartesian chart.

>>> import coordinax.charts as cxc
>>> cxc.cartnd.cartesian is cxc.cartnd
True

"""
