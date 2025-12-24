"""Representation of coordinates in different systems."""

__all__: tuple[str, ...] = ()

from jaxtyping import Real, Shaped
from typing import TypeAlias

from astropy.units import (
    CompositeUnit as AstropyCompositeUnit,
    Unit as AstropyUnit,
    UnitBase as AstropyUnitBase,
)

import unxt as u

Shape: TypeAlias = tuple[int, ...]
Unit: TypeAlias = AstropyUnit | AstropyUnitBase | AstropyCompositeUnit

BBtScalarQ = Shaped[u.AbstractQuantity, "*#batch"]

ScalarTime = Real[u.Q["time"], ""]

BBtTime = Real[u.Q["time"], "*#batch"]

BBtArea = Real[u.Q["area"], "*#batch"]
BBtKinematicFlux = Real[u.Q["diffusivity"], "*#batch"]
BBtSpecificEnergy = Real[u.Q["specific energy"], "*#batch"]

BBtLength = Real[u.Q["length"], "*#batch"]

BBtSpeed = Real[u.Q["speed"], "*#batch"]
BBtAngularSpeed = Real[u.Q["angular speed"], "*#batch"]

BBtAcc = Real[u.Q["acceleration"], "*#batch"]
BBtAngularAcc = Real[u.Q["angular acceleration"], "*#batch"]


TimeBatchOrScalar = ScalarTime | BBtTime
