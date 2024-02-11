"""Representation of coordinates in different systems."""

__all__: list[str] = []

from jax_quantity import Quantity
from jaxtyping import Float, Shaped

FloatScalarQ = Float[Quantity, ""]
BatchFloatScalarQ = Shaped[FloatScalarQ, "*batch"]
