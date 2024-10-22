"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.numpy as xp
from dataclassish import field_values
from unxt import AbstractQuantity, Quantity, UncheckedQuantity

from .base_pos import AbstractPos
from coordinax._src.distance import Distance
from coordinax._src.utils import full_shaped

#####################################################################
# Convert to Quantity


@conversion_method(type_from=AbstractPos, type_to=AbstractQuantity)  # type: ignore[misc]
def convert_pos_to_absquantity(obj: AbstractPos, /) -> AbstractQuantity:
    """`coordinax.AbstractPos` -> `unxt.AbstractQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    cart = full_shaped(obj.represent_as(obj._cartesian_cls))  # noqa: SLF001
    return xp.stack(tuple(field_values(cart)), axis=-1)


@conversion_method(type_from=AbstractPos, type_to=Quantity)  # type: ignore[misc]
def convert_pos_to_q(obj: AbstractPos, /) -> Quantity["length"]:
    """`coordinax.AbstractPos` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Quantity)


@conversion_method(type_from=AbstractPos, type_to=UncheckedQuantity)  # type: ignore[misc]
def convert_pos_to_uncheckedq(
    obj: AbstractPos, /
) -> Shaped[UncheckedQuantity, "*batch 1"]:
    """`coordinax.AbstractPos` -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity, UncheckedQuantity

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), UncheckedQuantity)


@conversion_method(type_from=AbstractPos, type_to=Distance)  # type: ignore[misc]
def convert_pos_to_distance(obj: AbstractPos, /) -> Shaped[Distance, "*batch 1"]:
    """`coordinax.AbstractPos` -> `coordinax.Distance`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import AbstractQuantity, Quantity
    >>> from coordinax.distance import Distance

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Distance)
