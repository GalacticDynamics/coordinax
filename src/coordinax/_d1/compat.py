"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractPosition1D
from .cartesian import CartesianAcceleration1D, CartesianPosition1D, CartesianVelocity1D
from coordinax._utils import dataclass_values, full_shaped


@conversion_method(type_from=AbstractPosition1D, type_to=Quantity)  # type: ignore[misc]
def vec_to_q(obj: AbstractPosition1D, /) -> Shaped[Quantity["length"], "*batch 1"]:
    """`coordinax.AbstractPosition1D` -> `unxt.Quantity`."""
    cart = full_shaped(obj.represent_as(CartesianPosition1D))
    return xp.stack(tuple(dataclass_values(cart)), axis=-1)


@conversion_method(type_from=CartesianAcceleration1D, type_to=Quantity)  # type: ignore[misc]
@conversion_method(type_from=CartesianVelocity1D, type_to=Quantity)  # type: ignore[misc]
def vec_diff_to_q(
    obj: CartesianVelocity1D | CartesianAcceleration1D, /
) -> Shaped[Quantity["speed"], "*batch 1"]:
    """`coordinax.CartesianVelocity1D` -> `unxt.Quantity`."""
    return xp.stack(tuple(dataclass_values(full_shaped(obj))), axis=-1)
