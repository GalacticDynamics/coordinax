"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractPosition2D
from .cartesian import CartesianPosition2D, CartesianVelocity2D
from coordinax._utils import dataclass_values, full_shaped

#####################################################################
# Quantity


@conversion_method(type_from=AbstractPosition2D, type_to=Quantity)  # type: ignore[misc]
def vec_to_q(obj: AbstractPosition2D, /) -> Shaped[Quantity["length"], "*batch 2"]:
    """`coordinax.AbstractPosition2D` -> `unxt.Quantity`."""
    cart = full_shaped(obj.represent_as(CartesianPosition2D))
    return xp.stack(tuple(dataclass_values(cart)), axis=-1)


@conversion_method(type_from=CartesianVelocity2D, type_to=Quantity)  # type: ignore[misc]
def vec_diff_to_q(obj: CartesianVelocity2D, /) -> Shaped[Quantity["speed"], "*batch 2"]:
    """`coordinax.CartesianVelocity2D` -> `unxt.Quantity`."""
    return xp.stack(tuple(dataclass_values(full_shaped(obj))), axis=-1)
