"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from dataclassish import field_values
from unxt import AbstractQuantity, Distance, Quantity, UncheckedQuantity

from .base_pos import AbstractPosition
from coordinax._coordinax.utils import full_shaped

#####################################################################
# Convert to Quantity


@conversion_method(type_from=AbstractPosition, type_to=AbstractQuantity)  # type: ignore[misc]
def convert_pos_to_absquantity(obj: AbstractPosition, /) -> AbstractQuantity:
    """`coordinax.AbstractPosition` -> `unxt.AbstractQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity

    >>> pos = cx.CartesianPosition1D.constructor([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPosition.constructor([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPosition(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition3D.constructor([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    cart = full_shaped(obj.represent_as(obj._cartesian_cls))  # noqa: SLF001
    return xp.stack(tuple(field_values(cart)), axis=-1)


@conversion_method(type_from=AbstractPosition, type_to=Quantity)  # type: ignore[misc]
def convert_pos_to_q(obj: AbstractPosition, /) -> Quantity["length"]:
    """`coordinax.AbstractPosition` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity

    >>> pos = cx.CartesianPosition1D.constructor([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPosition.constructor([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPosition(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition3D.constructor([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Quantity)


@conversion_method(type_from=AbstractPosition, type_to=UncheckedQuantity)  # type: ignore[misc]
def convert_pos_to_uncheckedq(
    obj: AbstractPosition, /
) -> Shaped[UncheckedQuantity, "*batch 1"]:
    """`coordinax.AbstractPosition` -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity, UncheckedQuantity

    >>> pos = cx.CartesianPosition1D.constructor([1.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPosition.constructor([1.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPosition(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition3D.constructor([1.0, 2.0, 3.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), UncheckedQuantity)


@conversion_method(type_from=AbstractPosition, type_to=Distance)  # type: ignore[misc]
def convert_pos_to_distance(obj: AbstractPosition, /) -> Shaped[Distance, "*batch 1"]:
    """`coordinax.AbstractPosition` -> `unxt.Distance`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity, Distance

    >>> pos = cx.CartesianPosition1D.constructor([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPosition.constructor([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPosition(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPosition3D.constructor([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPosition(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Distance)
