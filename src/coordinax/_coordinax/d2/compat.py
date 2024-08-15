"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from dataclassish import field_values
from unxt import AbstractQuantity, Quantity

from .base import AbstractPosition2D
from .cartesian import CartesianAcceleration2D, CartesianPosition2D, CartesianVelocity2D
from coordinax._coordinax.operators.base import AbstractOperator, op_call_dispatch
from coordinax._coordinax.typing import TimeBatchOrScalar
from coordinax._coordinax.utils import full_shaped

#####################################################################
# Convert to Quantity


@conversion_method(type_from=AbstractPosition2D, type_to=Quantity)  # type: ignore[misc]
def vec_to_q(obj: AbstractPosition2D, /) -> Shaped[Quantity["length"], "*batch 2"]:
    """`coordinax.AbstractPosition2D` -> `unxt.Quantity`."""
    cart = full_shaped(obj.represent_as(CartesianPosition2D))
    return xp.stack(tuple(field_values(cart)), axis=-1)


@conversion_method(type_from=CartesianAcceleration2D, type_to=Quantity)  # type: ignore[misc]
@conversion_method(type_from=CartesianVelocity2D, type_to=Quantity)  # type: ignore[misc]
def vec_diff_to_q(obj: CartesianVelocity2D, /) -> Shaped[AbstractQuantity, "*batch 2"]:
    """`coordinax.CartesianVelocity2D` -> `unxt.Quantity`."""
    return xp.stack(tuple(field_values(full_shaped(obj))), axis=-1)


#####################################################################
# Operators

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
