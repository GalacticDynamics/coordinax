# pylint: disable=duplicate-code
"""3-dimensional."""

__all__: list[str] = []

from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

from unxt import Quantity

from .cartesian import CartesianPosition2D
from coordinax._coordinax.operators._base import AbstractOperator, op_call_dispatch
from coordinax._coordinax.typing import TimeBatchOrScalar

Q2: TypeAlias = Shaped[Quantity["length"], "*#batch 2"]


@op_call_dispatch
def call(self: AbstractOperator, x: Q2, /) -> Q2:
    """Dispatch to the operator's `__call__` method."""
    return self(CartesianPosition2D.constructor(x))


@op_call_dispatch
def call(
    self: AbstractOperator, x: Q2, t: TimeBatchOrScalar, /
) -> tuple[Q2, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method."""
    vec, t = self(CartesianPosition2D.constructor(x), t)
    return convert(vec, Quantity), t
