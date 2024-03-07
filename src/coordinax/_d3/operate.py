# pylint: disable=duplicate-code
"""3-dimensional."""

__all__: list[str] = []

from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

from jax_quantity import Quantity

from .builtin import Cartesian3DVector
from coordinax._typing import TimeBatchOrScalar
from coordinax.operators._base import AbstractOperator, op_call_dispatch

Q3: TypeAlias = Shaped[Quantity["length"], "*#batch 3"]


@op_call_dispatch
def call(self: AbstractOperator, x: Q3, /) -> Q3:
    """Dispatch to the operator's `__call__` method."""
    return self(Cartesian3DVector.constructor(x))


@op_call_dispatch
def call(
    self: AbstractOperator, x: Q3, t: TimeBatchOrScalar, /
) -> tuple[Q3, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method."""
    vec, t = self(Cartesian3DVector.constructor(x), t)
    return convert(vec, Quantity), t
