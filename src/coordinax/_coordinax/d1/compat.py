"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from dataclassish import field_values
from unxt import AbstractQuantity, Quantity

from .cartesian import CartesianAcceleration1D, CartesianPosition1D, CartesianVelocity1D
from coordinax._coordinax.operators.base import AbstractOperator, op_call_dispatch
from coordinax._coordinax.typing import TimeBatchOrScalar
from coordinax._coordinax.utils import full_shaped

#####################################################################
# Convert to Quantity


@conversion_method(type_from=CartesianAcceleration1D, type_to=AbstractQuantity)  # type: ignore[misc]
@conversion_method(type_from=CartesianVelocity1D, type_to=AbstractQuantity)  # type: ignore[misc]
def vec_diff_to_q(
    obj: CartesianVelocity1D | CartesianAcceleration1D, /
) -> Shaped[AbstractQuantity, "*batch 1"]:
    """`coordinax.CartesianVelocity1D` -> `unxt.Quantity`."""
    return xp.stack(tuple(field_values(full_shaped(obj))), axis=-1)


#####################################################################
# Operators


Q1: TypeAlias = Shaped[Quantity["length"], "*#batch 1"]


@op_call_dispatch
def call(self: AbstractOperator, x: Q1, /) -> Q1:
    """Dispatch to the operator's `__call__` method."""
    return self(CartesianPosition1D.constructor(x))


@op_call_dispatch
def call(
    self: AbstractOperator, x: Q1, t: TimeBatchOrScalar, /
) -> tuple[Q1, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method."""
    vec, t = self(CartesianPosition1D.constructor(x), t)
    return convert(vec, Quantity), t
