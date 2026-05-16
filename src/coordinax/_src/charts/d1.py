"""1-D charts.

>>> import coordinax.charts as cxc
>>> cxc.cart1d.components
('x',)

"""

__all__ = (
    "Abstract1D",
    "Cart1D",
    "cart1d",
    "Radial1D",
    "radial1d",
    "Time1D",
    "time1d",
)

import dataclasses

from typing import Any, Final, Literal as L, TypeVar, final  # noqa: N817
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
from coordinax._src.euclidean.atlas import (
    EUCLIDEAN_ATLAS_DEFAULT_CHARTS,
    EuclideanAtlas,
)
from coordinax._src.euclidean.manifold import R1

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # ty: ignore[invalid-type-form]
V = TypeVar("V")


class Abstract1D(AbstractDimensionalFlag, n=1):
    """Marker flag for 1D representations.

    A 1D representation has exactly one coordinate component. Examples include
    Cartesian $(x)$ or radial $(r)$ coordinates.
    """

    # TODO: add a check it's 1D

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

Cart1DKeys = tuple[L["x"]]
Cart1DDims = tuple[Len]


@jtu.register_static
@final
@chart_dataclass_decorator
class Cart1D(AbstractFixedComponentsChart[Cart1DKeys, Cart1DDims], Abstract1D):
    r"""One-dimensional Cartesian chart $(x)$.

    Components are ordered as ``("x",)`` with dimension ``("length",)``.

    This chart is the canonical 1D Cartesian chart and is returned by
    {func}`coordinax.charts.cartesian_chart` for 1D charts.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.cart1d.components
    ('x',)

    >>> cxc.cart1d.coord_dimensions
    ('length',)

    >>> isinstance(cxc.cartesian_chart(cxc.cart1d), cxc.Cart1D)
    True

    """

    _: dataclasses.KW_ONLY
    M: AbstractManifold = R1

    @override
    @property
    def cartesian(self) -> "Cart1D":
        """Return the canonical Cartesian chart for a 1D chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Cart1D().cartesian, cxc.Cart1D)
        True

        """
        return self


cart1d: Final = Cart1D()
"""The canonical 1D Cartesian chart.

>>> import coordinax.charts as cxc
>>> cxc.cart1d.cartesian is cxc.cart1d
True

"""

EUCLIDEAN_ATLAS_DEFAULT_CHARTS[1] = cart1d


# -----------------------------------------------
# Radial

RadialKeys = tuple[L["r"]]
Radial1DDims = tuple[Len]


@EuclideanAtlas.register
@jtu.register_static
@final
@chart_dataclass_decorator
class Radial1D(AbstractFixedComponentsChart[RadialKeys, Radial1DDims], Abstract1D):
    r"""One-dimensional radial chart $(r)$.

    Components are ordered as ``("r",)`` with dimension ``("length",)``.

    This chart is semantically equivalent to {class}`coordinax.charts.Cart1D`
    but uses ``r`` instead of ``x``. Its canonical Cartesian projection is
    {obj}`coordinax.charts.cart1d`.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.radial1d.components
    ('r',)

    >>> cxc.radial1d.coord_dimensions
    ('length',)

    >>> isinstance(cxc.cartesian_chart(cxc.radial1d), cxc.Cart1D)
    True

    """

    _: dataclasses.KW_ONLY
    M: AbstractManifold = R1

    @override
    @property
    def cartesian(self) -> "Cart1D":
        """Return the canonical Cartesian chart for a 1D radial chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Radial1D().cartesian, cxc.Cart1D)
        True

        """
        return Cart1D(M=self.M)


radial1d: Final = Radial1D()
"""The canonical 1D radial chart.

>>> import coordinax.charts as cxc
>>> cxc.radial1d.cartesian is cxc.cart1d
True

"""

# -----------------------------------------------
# Time

TimeKeys = tuple[L["t"]]
TimeDims = tuple[L["time"]]


@EuclideanAtlas.register
@jtu.register_static
@final
@chart_dataclass_decorator
class Time1D(AbstractFixedComponentsChart[TimeKeys, TimeDims], Abstract1D):
    """One-dimensional time chart ``(t)``.

    Components are ordered as ``("t",)`` with dimension ``("time",)``.

    This chart is the canonical 1D time chart and is often used as the first
    factor in spacetime product charts.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> cxc.time1d.components
    ('t',)

    >>> cxc.time1d.coord_dimensions
    ('time',)

    >>> isinstance(cxc.cartesian_chart(cxc.time1d), cxc.Time1D)
    True

    """

    _: dataclasses.KW_ONLY
    M: AbstractManifold = R1

    @override
    @property
    def cartesian(self) -> "Time1D":
        """Return the canonical Cartesian chart for a 1D time chart.

        >>> import coordinax.charts as cxc
        >>> isinstance(cxc.Time1D().cartesian, cxc.Time1D)
        True

        """
        return time1d


time1d: Final = Time1D()
"""The canonical 1D time chart.

>>> import coordinax.charts as cxc
>>> cxc.time1d.cartesian is cxc.time1d
True

"""
