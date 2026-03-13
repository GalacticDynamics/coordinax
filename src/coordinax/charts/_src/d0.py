"""0-D representation."""

__all__ = ("Abstract0D", "Cart0D", "cart0d")


from typing import Any, Final, Literal as L, final  # noqa: N817
from typing_extensions import override

import plum

from .base import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
)
from .utils import is_not_abstract_chart_subclass


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


@final
@chart_dataclass_decorator
class Cart0D(AbstractFixedComponentsChart[ZeroDKeys, ZeroDDims], Abstract0D):
    """0D position representation."""


cart0d: Final = Cart0D()


@plum.dispatch
def cartesian_chart(obj: Cart0D, /) -> Cart0D:
    return cart0d
