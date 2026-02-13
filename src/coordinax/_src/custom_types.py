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
    "OptUSys",
    # Array-related
    "Shape",
    "BatchableAngleQ",
    "Ks",
    "Ds",
    "HasShape",
)

from jaxtyping import Real, Shaped
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable
from typing_extensions import TypeVar

import unxt as u

from coordinax.api import ComponentsKey

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
OptUSys: TypeAlias = u.AbstractUnitSystem | None

# =========================================================
# Array-related Types

Shape: TypeAlias = tuple[int, ...]

# Shaped Arrays
BBtScalarQ = Shaped[u.AbstractQuantity, "*#batch"]

ScalarTime = Real[u.Q["time"], ""]  # type: ignore[type-arg]

BBtTime = Real[u.Q["time"], "*#batch"]  # type: ignore[type-arg]

BBtArea = Real[u.Q["area"], "*#batch"]  # type: ignore[type-arg]
BBtKinematicFlux = Real[u.Q["diffusivity"], "*#batch"]  # type: ignore[type-arg]
BBtSpecificEnergy = Real[u.Q["specific energy"], "*#batch"]  # type: ignore[type-arg,valid-type]

BBtLength = Real[u.Q["length"], "*#batch"]  # type: ignore[type-arg]

BBtSpeed = Real[u.Q["speed"], "*#batch"]  # type: ignore[type-arg]
BBtAngularSpeed = Real[u.Q["angular speed"], "*#batch"]  # type: ignore[type-arg,valid-type]

BBtAcc = Real[u.Q["acceleration"], "*#batch"]  # type: ignore[type-arg]
BBtAngularAcc = Real[u.Q["angular acceleration"], "*#batch"]  # type: ignore[type-arg,valid-type]


TimeBatchOrScalar = ScalarTime | BBtTime


#: Batchable angular-type Quantity.
BatchableAngularQuantity = Shaped[u.Q["angle"], "*#batch"]  # type: ignore[type-arg]

#: Batchable Angle.
BatchableAngle = Shaped[u.quantity.AbstractAngle, "*#batch"]

#: Batchable Angle or Angular Quantity.
BatchableAngleQ = BatchableAngle | BatchableAngularQuantity


# =========================================================
# Vector-related Types

# Component Value Type
V = TypeVar("V", default=Any)

Ks = TypeVar("Ks", bound=tuple[ComponentsKey, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])


@runtime_checkable
class HasShape(Protocol):
    """A protocol for objects that have a shape attribute."""

    @property
    def shape(self) -> Shape:
        """The shape of the object."""
        raise NotImplementedError  # pragma: no cover
