"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import TypeAlias

from astropy.units import (
    CompositeUnit as AstropyCompositeUnit,
    Unit as AstropyUnit,
    UnitBase as AstropyUnitBase,
)
from jaxtyping import Float, Int, Shaped

from unxt.quantity import AbstractQuantity, Quantity

Unit: TypeAlias = AstropyUnit | AstropyUnitBase | AstropyCompositeUnit

BatchableScalarQ = Shaped[AbstractQuantity, "*#batch"]

FloatScalarQ = Float[AbstractQuantity, ""]
BatchFloatScalarQ = Shaped[FloatScalarQ, "*batch"]
BatchableFloatScalarQ = Shaped[FloatScalarQ, "*#batch"]

ScalarTime = Float[Quantity["time"], ""] | Int[Quantity["time"], ""]

BatchableTime = Shaped[Quantity["time"], "*#batch"]

BatchableArea = Shaped[Quantity["area"], "*#batch"]
BatchableDiffusivity = Shaped[Quantity["diffusivity"], "*#batch"]
BatchableSpecificEnergy = Shaped[Quantity["specific energy"], "*#batch"]

BatchableSpeed = Shaped[Quantity["speed"], "*#batch"]
BatchableAngularSpeed = Shaped[Quantity["angular speed"], "*#batch"]

BatchableAcc = Shaped[Quantity["acceleration"], "*#batch"]
BatchableAngularAcc = Shaped[Quantity["angular acceleration"], "*#batch"]


TimeBatchOrScalar = ScalarTime | BatchableTime
