"""4-Dimensional charts."""

__all__ = ("Abstract4D",)

from typing import Any, Literal as L  # noqa: N817
from typing_extensions import override

from coordinax._src.base import (
    AbstractDimensionalFlag,
    is_not_abstract_chart_subclass,
)


class Abstract4D(AbstractDimensionalFlag, n=4):
    """Marker flag for 4-D representations.

    A 4-D representation has exactly four coordinate components. The primary
    example is the Minkowski spacetime chart ``(ct, x, y, z)``.
    """

    # TODO: add a check it's 4D

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
