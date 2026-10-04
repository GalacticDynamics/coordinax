"""Representation of coordinates in different systems."""

__all__: tuple[str, ...] = ("DimQuantity",)

from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from astropy.units import (
    CompositeUnit as AstropyCompositeUnit,
    Unit as AstropyUnit,
    UnitBase as AstropyUnitBase,
)
from jaxtyping import Real, Shaped

import unxt as u

if TYPE_CHECKING:
    # mypy does not see unxt, so this is `Any`, as `u.Quantity["length"]` was.
    from unxt import Quantity as DimQuantity
else:

    class _DimQuantityMeta(type(u.Quantity)):
        """Metaclass of `DimQuantity`: instance checks by physical dimension."""

        def __instancecheck__(cls, obj: Any, /) -> bool:
            return isinstance(obj, u.Quantity) and (
                cls.dimension is None or u.dimension_of(obj) == cls.dimension
            )

        @cache  # noqa: B019  # pylint: disable=method-cache-max-size-none
        def __getitem__(cls, dim: Any, /) -> "type[DimQuantity]":
            dim = u.dimension(dim)
            return _DimQuantityMeta(
                f"DimQuantity[{dim}]",
                (cls,),
                {"dimension": dim, "__module__": __name__},
            )

    class DimQuantity(u.Quantity, metaclass=_DimQuantityMeta):
        """A `unxt.Quantity` annotation that carries a physical dimension.

        A stop-gap for the unxt v2 port: ``u.Quantity["length"]`` is a no-op in
        v2. Prefer `u.Q`. Use this only where the dimension is load-bearing:

        - vector field annotations, which `AbstractVector.dimensions` reads; and
        - overloads that differ only by dimension, e.g. ``op(q, p)`` vs
          ``op(t, x)``. Its `isinstance` checks the dimension.

        As in v1, other quantity types (e.g. `unxt.Angle`) are not instances.
        Never instantiated; it subclasses `unxt.Quantity` only so plum ranks it
        as narrower.

        >>> import unxt as u
        >>> isinstance(u.Quantity(1, "km"), DimQuantity["length"])
        True
        >>> isinstance(u.Quantity(1, "s"), DimQuantity["length"])
        False
        >>> u.dimension_of(DimQuantity["speed"])
        PhysicalType({'speed', 'velocity'})

        """

        dimension: ClassVar[Any] = None
        __faithful__: ClassVar[bool] = False  # plum: `isinstance` depends on the unit

    @u.dimension_of.dispatch
    def dimension_of(obj: _DimQuantityMeta, /) -> u.dims.AbstractDimension:
        return obj.dimension


Shape: TypeAlias = tuple[int, ...]
Unit: TypeAlias = AstropyUnit | AstropyUnitBase | AstropyCompositeUnit

BBtScalarQ = Shaped[u.AbstractQuantity, "*#batch"]

ScalarTime = Real[DimQuantity["time"], ""]

BBtTime = Real[DimQuantity["time"], "*#batch"]

BBtArea = Real[DimQuantity["area"], "*#batch"]
BBtKinematicFlux = Real[DimQuantity["diffusivity"], "*#batch"]
BBtSpecificEnergy = Real[DimQuantity["specific energy"], "*#batch"]

BBtLength = Real[DimQuantity["length"], "*#batch"]

BBtSpeed = Real[DimQuantity["speed"], "*#batch"]
BBtAngularSpeed = Real[DimQuantity["angular speed"], "*#batch"]

BBtAcc = Real[DimQuantity["acceleration"], "*#batch"]
BBtAngularAcc = Real[DimQuantity["angular acceleration"], "*#batch"]


TimeBatchOrScalar = ScalarTime | BBtTime
