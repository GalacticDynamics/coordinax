"""4-Dimensional charts."""

__all__ = ("Abstract4D",)

from typing import Any, Literal as L, override  # noqa: N817

from coordinax._src.base import (
    AbstractDimensionalFlag,
)


class Abstract4D(AbstractDimensionalFlag, n=4):
    """Marker flag for 4-D representations.

    A 4-D representation has exactly four coordinate components. The primary
    example is the Minkowski spacetime chart ``(ct, x, y, z)``.
    """

    @override
    def __init_subclass__(cls, n: int | L["N"] | None = None, **kw: Any) -> None:
        if n is not None:
            msg = f"{cls.__name__} does not support variable n"
            raise NotImplementedError(msg)
        super().__init_subclass__(n=n, **kw)
