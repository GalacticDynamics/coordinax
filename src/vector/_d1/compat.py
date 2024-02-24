"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


import array_api_jax_compat as xp
from jax_quantity import Quantity
from plum import conversion_method

from vector._utils import dataclass_values

from .base import Abstract1DVector
from .builtin import Cartesian1DVector, CartesianDifferential1D

#####################################################################
# Quantity


@conversion_method(type_from=Abstract1DVector, type_to=Quantity)  # type: ignore[misc]
def vec_to_q(obj: Abstract1DVector, /) -> Quantity["length"]:
    """`vector.Abstract1DVector` -> `jax_quantity.Quantity`."""
    cart = full_shaped(obj.represent_as(Cartesian1DVector))
    return xp.stack(tuple(dataclass_values(cart)), axis=-1)


@conversion_method(type_from=CartesianDifferential1D, type_to=Quantity)  # type: ignore[misc]
def vec_diff_to_q(obj: CartesianDifferential1D, /) -> Quantity["speed"]:
    """`vector.CartesianDifferential1D` -> `jax_quantity.Quantity`."""
    return xp.stack(tuple(dataclass_values(full_shaped(obj))), axis=-1)
