"""Vector."""

__all__ = ("Abstract2D", "Cart2D", "cart2d", "Polar2D", "polar2d")


from typing import Any, Final, Literal as L, final  # noqa: N817
from typing_extensions import override

import plum

from .base import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
)
from .utils import is_not_abstract_chart_subclass
from coordinax.internal.custom_types import Ang, Len


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


@final
@chart_dataclass_decorator
class Cart2D(AbstractFixedComponentsChart[Cart2DKeys, Cart2DDims], Abstract2D):
    pass


cart2d: Final = Cart2D()


@plum.dispatch
def cartesian_chart(obj: Cart2D, /) -> Cart2D:
    return cart2d


# -----------------------------------------------
# Polar

PolarKeys = tuple[L["r"], L["theta"]]
Polar2DDims = tuple[Len, Ang]


@final
@chart_dataclass_decorator
class Polar2D(AbstractFixedComponentsChart[PolarKeys, Polar2DDims], Abstract2D):
    pass


polar2d: Final = Polar2D()


@plum.dispatch
def cartesian_chart(obj: Polar2D, /) -> Cart2D:
    return cart2d
