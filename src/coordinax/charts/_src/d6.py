"""Vector."""

__all__ = ("Abstract6D", "PoincarePolar6D", "poincarepolar6d")


from typing import Any, Final, Literal as L, final  # noqa: N817
from typing_extensions import override

from .base import (
    AbstractDimensionalFlag,
    AbstractFixedComponentsChart,
    chart_dataclass_decorator,
)
from .utils import is_not_abstract_chart_subclass
from coordinax.internal.custom_types import Len, Spd


class Abstract6D(AbstractDimensionalFlag, n=6):
    """Marker flag for 6-D representations.

    An 6-D representation has an arbitrary number of coordinate components.
    Examples include Cartesian representations in arbitrary dimensions.
    """

    # TODO: add a check it's 6D

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        # Enforce that this is a subclass of AbstractChart
        if is_not_abstract_chart_subclass(cls):
            msg = f"{cls.__name__} must be a subclass of AbstractChart"
            raise TypeError(msg)
        # n is already fixed to 6
        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)


PoincarePolarKeys = tuple[
    L["rho"], L["pp_phi"], L["z"], L["dt_rho"], L["dt_pp_phi"], L["dt_z"]
]

PoincarePolarDims = tuple[
    Len, L["length / time**0.5"], Len, Spd, L["length / time**1.5"], Spd
]


@final
@chart_dataclass_decorator
class PoincarePolar6D(
    AbstractFixedComponentsChart[PoincarePolarKeys, PoincarePolarDims], Abstract6D
):
    pass


poincarepolar6d: Final = PoincarePolar6D()
