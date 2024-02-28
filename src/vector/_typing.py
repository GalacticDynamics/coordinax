"""Representation of coordinates in different systems."""

__all__: list[str] = []

from jaxtyping import Float, Shaped

from jax_quantity import Quantity

FloatScalarQ = Float[Quantity, ""]
BatchFloatScalarQ = Shaped[FloatScalarQ, "*batch"]
BatchableFloatScalarQ = Shaped[FloatScalarQ, "*#batch"]

BatchableAngle = Shaped[Quantity["angle"], "*#batch"]
BatchableLength = Shaped[Quantity["length"], "*#batch"]
BatchableSpeed = Shaped[Quantity["speed"], "*#batch"]
BatchableTime = Shaped[Quantity["time"], "*#batch"]
BatchableAngularSpeed = Shaped[Quantity["angular speed"], "*#batch"]
