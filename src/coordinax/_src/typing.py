"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import TypeAlias

import astropy.units as u
from jaxtyping import Float, Int, Shaped

from unxt import AbstractQuantity, Quantity

from .distance.base import AbstractDistance

Unit: TypeAlias = u.Unit | u.UnitBase | u.CompositeUnit

FloatScalarQ = Float[AbstractQuantity, ""]
BatchFloatScalarQ = Shaped[FloatScalarQ, "*batch"]
BatchableFloatScalarQ = Shaped[FloatScalarQ, "*#batch"]

ScalarTime = Float[Quantity["time"], ""] | Int[Quantity["time"], ""]

BatchableAngle = Shaped[Quantity["angle"], "*#batch"]
BatchableLength = Shaped[Quantity["length"], "*#batch"]
BatchableDistance = Shaped[AbstractDistance, "*#batch"]
BatchableTime = Shaped[Quantity["time"], "*#batch"]

BatchableSpeed = Shaped[Quantity["speed"], "*#batch"]
BatchableAngularSpeed = Shaped[Quantity["angular speed"], "*#batch"]

BatchableAcc = Shaped[Quantity["acceleration"], "*#batch"]
BatchableAngularAcc = Shaped[Quantity["angular acceleration"], "*#batch"]


TimeBatchOrScalar = ScalarTime | BatchableTime
