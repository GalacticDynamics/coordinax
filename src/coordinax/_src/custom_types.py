"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import TypeAlias

from astropy.units import (
    CompositeUnit as AstropyCompositeUnit,
    Unit as AstropyUnit,
    UnitBase as AstropyUnitBase,
)
from jaxtyping import Real, Shaped

import unxt as u

Shape: TypeAlias = tuple[int, ...]
Unit: TypeAlias = AstropyUnit | AstropyUnitBase | AstropyCompositeUnit

BBtScalarQ = Shaped[u.AbstractQuantity, "*#batch"]

ScalarTime = Real[u.Quantity["time"], ""]

BBtTime = Real[u.Quantity["time"], "*#batch"]

BBtArea = Real[u.Quantity["area"], "*#batch"]
BBtKinematicFlux = Real[u.Quantity["diffusivity"], "*#batch"]
BBtSpecificEnergy = Real[u.Quantity["specific energy"], "*#batch"]

BBtLength = Real[u.Quantity["length"], "*#batch"]

BBtSpeed = Real[u.Quantity["speed"], "*#batch"]
BBtAngularSpeed = Real[u.Quantity["angular speed"], "*#batch"]

BBtAcc = Real[u.Quantity["acceleration"], "*#batch"]
BBtAngularAcc = Real[u.Quantity["angular acceleration"], "*#batch"]


TimeBatchOrScalar = ScalarTime | BBtTime
