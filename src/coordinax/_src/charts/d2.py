"""2-Dimensional charts."""

__all__ = ("Abstract2D", "Cart2D", "cart2d", "Polar2D", "polar2d")


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
from coordinax._src.base_topo import AbstractTopologicalManifold
from coordinax._src.custom_types import Ang, Len
from coordinax._src.euclidean.atlas import (
    EUCLIDEAN_ATLAS_DEFAULT_CHARTS,
    EuclideanAtlas,
)
from coordinax._src.euclidean.manifold import euclidean2d


class Abstract2D(AbstractDimensionalFlag, n=2):
    """Marker flag for 2D representations.

    A 2D representation has exactly two coordinate components. This does not
    imply that the underlying manifold is flat; for example, the two-sphere uses
    two angular coordinates but represents a curved surface.
    """

    # TODO: add a check it's 2D

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
# Cartesian

Cart2DKeys = tuple[L["x"], L["y"]]
Cart2DDims = tuple[Len, Len]


@EuclideanAtlas.register
@jtu.register_static
@final
@chart_dataclass_decorator
class Cart2D(AbstractFixedComponentsChart[Cart2DKeys, Cart2DDims], Abstract2D):
    r"""Two-dimensional Cartesian chart $(x, y)$.

    Components are ordered as ``("x", "y")`` with dimensions ``("length",
    "length")``.

    This chart is the canonical 2D Cartesian chart and is returned by
    {func}`coordinax.charts.cartesian_chart` for 2D charts.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.cart2d.components
    ('x', 'y')

    >>> cxc.cart2d.coord_dimensions
    ('length', 'length')

    >>> isinstance(cxc.cartesian_chart(cxc.cart2d), cxc.Cart2D)
    True

    """

    _: dataclasses.KW_ONLY
    M: AbstractTopologicalManifold = euclidean2d

    @override
    @property
    def cartesian(self) -> "Cart2D":
        """Return the canonical Cartesian chart for a 2D chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Cart2D().cartesian, cxc.Cart2D)
        True

        """
        return self


cart2d: Final = Cart2D()
"""The canonical 2D Cartesian chart.

>>> import coordinax.charts as cxc
>>> cxc.cart2d.cartesian is cxc.cart2d
True

"""

EUCLIDEAN_ATLAS_DEFAULT_CHARTS[2] = cart2d


# -----------------------------------------------
# Polar

PolarKeys = tuple[L["r"], L["theta"]]
Polar2DDims = tuple[Len, Ang]


@EuclideanAtlas.register
@jtu.register_static
@final
@chart_dataclass_decorator
class Polar2D(AbstractFixedComponentsChart[PolarKeys, Polar2DDims], Abstract2D):
    r"""Two-dimensional polar chart $(r, \theta)$.

    Components are ordered as ``("r", "theta")`` with dimensions ``("length",
    "angle")``.

    This chart has direct transitions with {class}`coordinax.charts.Cart2D` and
    its canonical Cartesian projection is {obj}`coordinax.charts.cart2d`.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.polar2d.components
    ('r', 'theta')

    >>> cxc.polar2d.coord_dimensions
    ('length', 'angle')

    >>> isinstance(cxc.cartesian_chart(cxc.polar2d), cxc.Cart2D)
    True

    """

    _: dataclasses.KW_ONLY
    M: AbstractTopologicalManifold = euclidean2d

    @override
    @property
    def cartesian(self) -> "Cart2D":
        """Return the canonical Cartesian chart for a 2D chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Cart2D().cartesian, cxc.Cart2D)
        True

        """
        return Cart2D(M=self.M)


polar2d: Final = Polar2D()
"""The canonical 2D polar chart.

>>> import coordinax.charts as cxc
>>> cxc.polar2d.cartesian is cxc.cart2d
True

"""
