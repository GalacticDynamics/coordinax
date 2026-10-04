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
            try:
                dim = u.dimension(dim)
            except ValueError:  # e.g. "mag": not a physical type, but a unit
                # ponytail: "mag" -> 'unknown', so it admits any unknown-dimension
                # quantity; check unit convertibility if that ever matters.
                dim = u.dimension_of(u.unit(dim))
            return _DimQuantityMeta(
                f"DimQuantity[{dim}]",
                (cls,),
                {"dimension": dim, "__module__": __name__},
            )

    class DimQuantity(u.Quantity, metaclass=_DimQuantityMeta):
        """Annotation for a quantity of a given physical dimension.

        In unxt v2 ``u.Quantity["length"]`` is a no-op alias of `unxt.Quantity`.
        This restores what coordinax relies on from the v1 parametric class:
        `isinstance` (so plum dispatch and runtime type-checking) by dimension,
        `unxt.dimension_of` on the annotation, and a dimension-checking ``from_``.
        As in v1, other quantity types (e.g. `unxt.Angle`, `coordinax.Distance`)
        are not instances. Values are plain `unxt.Quantity`; this class is never
        instantiated, and subclasses `unxt.Quantity` only so plum ranks it as
        narrower.

        >>> import unxt as u
        >>> isinstance(u.Quantity(1, "km"), DimQuantity["length"])
        True
        >>> isinstance(u.Quantity(1, "s"), DimQuantity["length"])
        False
        >>> u.dimension_of(DimQuantity["speed"])
        PhysicalType({'speed', 'velocity'})
        >>> DimQuantity["length"].from_([1, 2], "km")
        Quantity(Array([1, 2], dtype=int32), unit='km')
        >>> try: DimQuantity["length"].from_(1, "s")
        ... except ValueError as e: print(e)
        Expected a quantity of dimension 'length', got 'time'.

        """

        dimension: ClassVar[Any] = None
        __faithful__: ClassVar[bool] = False  # plum: `isinstance` depends on the unit

        @classmethod
        def from_(cls, *args: Any, **kwargs: Any) -> u.Quantity:
            q = u.Quantity.from_(*args, **kwargs)
            if cls.dimension is not None and u.dimension_of(q) != cls.dimension:
                msg = (
                    f"Expected a quantity of dimension '{cls.dimension}', "
                    f"got '{u.dimension_of(q)}'."
                )
                raise ValueError(msg)
            return q

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
