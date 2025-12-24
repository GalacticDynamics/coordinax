"""Internal custom types for coordinax."""

__all__ = (
    # Dimension-related
    "DimensionLike",
    "Len",
    "Spd",
    "Acc",
    "Ang",
    "AngSpd",
    "AngAcc",
    # Units-related
    "Unit",
    # Array-related
    "Shape",
    "BatchableAngleQ",
    # Vector-related
    "ComponentKey",
    "ComponentsKey",
    "CDict",
    "CsDict",
    "Ks",
    "Ds",
)

from jaxtyping import Real, Shaped
from typing import Any, Literal, TypeAlias
from typing_extensions import TypeVar

import unxt as u

# Dimensions
DimensionLike: TypeAlias = u.AbstractDimension | str

#   Specific Dimensions
Len: TypeAlias = Literal["length"]
Spd: TypeAlias = Literal["speed"]
Acc: TypeAlias = Literal["acceleration"]
Ang: TypeAlias = Literal["angle"]
AngSpd: TypeAlias = Literal["angular speed"]
AngAcc: TypeAlias = Literal["angular acceleration"]


# Units
Unit: TypeAlias = u.AbstractUnit

# =========================================================
# Array-related Types

Shape: TypeAlias = tuple[int, ...]

# Shaped Arrays
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


#: Batchable angular-type Quantity.
BatchableAngularQuantity = Shaped[u.Q["angle"], "*#batch"]

#: Batchable Angle.
BatchableAngle = Shaped[u.quantity.AbstractAngle, "*#batch"]

#: Batchable Angle or Angular Quantity.
BatchableAngleQ = BatchableAngle | BatchableAngularQuantity


# =========================================================
# Vector-related Types

# Component key type: string for simple charts, tuple for product charts
ComponentKey: TypeAlias = str
ProductComponentKey: TypeAlias = tuple[str, str]
ComponentsKey: TypeAlias = ComponentKey | ProductComponentKey

# Component Value Type
V = TypeVar("V", default=Any)

# Parameter dictionary type alias (supports both flat and product keys)
CDict: TypeAlias = dict[ComponentKey, Any]
CsDict: TypeAlias = dict[ComponentsKey, Any]

Ks = TypeVar("Ks", bound=tuple[ComponentsKey, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
