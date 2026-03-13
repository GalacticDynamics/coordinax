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


from typing import Any, Final, Literal as L, TypeVar, final  # noqa: N817
from typing_extensions import override

import plum

from .base import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
)
from .utils import is_not_abstract_chart_subclass
from coordinax.internal.custom_types import Len

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # type: ignore[misc]
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


@final
@chart_dataclass_decorator
class Cart1D(AbstractFixedComponentsChart[Cart1DKeys, Cart1DDims], Abstract1D):
    pass


cart1d: Final = Cart1D()


@plum.dispatch
def cartesian_chart(obj: Cart1D, /) -> Cart1D:
    return cart1d


# -----------------------------------------------
# Radial

RadialKeys = tuple[L["r"]]
Radial1DDims = tuple[Len]


@final
@chart_dataclass_decorator
class Radial1D(AbstractFixedComponentsChart[RadialKeys, Radial1DDims], Abstract1D):
    pass


radial1d: Final = Radial1D()


@plum.dispatch
def cartesian_chart(obj: Radial1D, /) -> Cart1D:
    return cart1d


# -----------------------------------------------
# Time

TimeKeys = tuple[L["t"]]
TimeDims = tuple[L["time"]]


@final
@chart_dataclass_decorator
class Time1D(AbstractFixedComponentsChart[TimeKeys, TimeDims], Abstract1D):
    """1D time chart.

    A time chart has a single component "t" with time dimension.  This is the
    canonical 1D time chart used as the first factor in SpaceTimeCT product
    charts.

    """


time1d: Final = Time1D()


# Time1D is already Cartesian
@plum.dispatch
def cartesian_chart(obj: Time1D, /) -> Time1D:
    return time1d
