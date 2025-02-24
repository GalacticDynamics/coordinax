"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import TypeAlias

from astropy.units import (
    CompositeUnit as AstropyCompositeUnit,
    Unit as AstropyUnit,
    UnitBase as AstropyUnitBase,
)
from jaxtyping import Float, Int, Shaped

import unxt as u

Shape: TypeAlias = tuple[int, ...]
Unit: TypeAlias = AstropyUnit | AstropyUnitBase | AstropyCompositeUnit

BatchableScalarQ = Shaped[u.AbstractQuantity, "*#batch"]

FloatScalarQ = Float[u.AbstractQuantity, ""]
BatchFloatScalarQ = Shaped[FloatScalarQ, "*batch"]
BatchableFloatScalarQ = Shaped[FloatScalarQ, "*#batch"]

ScalarTime = Float[u.Quantity["time"], ""] | Int[u.Quantity["time"], ""]

BatchableTime = Shaped[u.Quantity["time"], "*#batch"]

BatchableArea = Shaped[u.Quantity["area"], "*#batch"]
BatchableDiffusivity = Shaped[u.Quantity["diffusivity"], "*#batch"]
BatchableSpecificEnergy = Shaped[u.Quantity["specific energy"], "*#batch"]

BatchableLength = Shaped[u.Quantity["length"], "*#batch"]

BatchableSpeed = Shaped[u.Quantity["speed"], "*#batch"]
BatchableAngularSpeed = Shaped[u.Quantity["angular speed"], "*#batch"]

BatchableAcc = Shaped[u.Quantity["acceleration"], "*#batch"]
BatchableAngularAcc = Shaped[u.Quantity["angular acceleration"], "*#batch"]


TimeBatchOrScalar = ScalarTime | BatchableTime
