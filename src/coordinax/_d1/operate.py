# pylint: disable=duplicate-code
"""3-dimensional."""

__all__: list[str] = []

from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

from unxt import Quantity

from .builtin import Cartesian1DVector
from coordinax._typing import TimeBatchOrScalar
from coordinax.operators._base import AbstractOperator, op_call_dispatch

Q1: TypeAlias = Shaped[Quantity["length"], "*#batch 1"]


@op_call_dispatch
def call(self: AbstractOperator, x: Q1, /) -> Q1:
    """Dispatch to the operator's `__call__` method."""
    return self(Cartesian1DVector.constructor(x))


@op_call_dispatch
def call(
    self: AbstractOperator, x: Q1, t: TimeBatchOrScalar, /
) -> tuple[Q1, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method."""
    vec, t = self(Cartesian1DVector.constructor(x), t)
    return convert(vec, Quantity), t
