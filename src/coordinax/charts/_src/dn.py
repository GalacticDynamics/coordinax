"""Vector."""

__all__ = ("AbstractND", "CartND", "cartnd")


from typing import Any, Final, Literal as L, final  # noqa: N817

import plum

from .base import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
)
from .utils import is_not_abstract_chart_subclass
from coordinax.internal.custom_types import Len


class AbstractND(AbstractDimensionalFlag, n="N"):
    """Marker flag for N-D representations.

    An N-D representation has an arbitrary number of coordinate components.
    Examples include Cartesian representations in arbitrary dimensions.
    """

    # TODO: add a check it's N-D

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


@final
@chart_dataclass_decorator
class CartND(AbstractFixedComponentsChart[CartNDKeys, CartNDDims], AbstractND):
    pass


cartnd: Final = CartND()


@plum.dispatch(precedence=-1)
def cartesian_chart(obj: CartND, /) -> CartND:
    return cartnd
