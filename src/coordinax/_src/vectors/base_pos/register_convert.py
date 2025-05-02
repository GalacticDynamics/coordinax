"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_values
from unxt.quantity import BareQuantity

from .core import AbstractPos
from coordinax._src.distances import Distance
from coordinax._src.vectors.utils import full_shaped

# ===================================================================
# Coordinax


@conversion_method(type_from=AbstractPos, type_to=Distance)  # type: ignore[arg-type,type-abstract]
def convert_pos_to_distance(obj: AbstractPos, /) -> Shaped[Distance, "*batch dims"]:
    """`coordinax.AbstractPos` -> `coordinax.Distance`.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> from coordinax.distance import Distance

    >>> pos = cx.vecs.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> convert(pos, Distance)
    Distance(Array([1, 2], dtype=int32), unit='km')

    >>> pos = cx.vecs.PolarPos(u.Quantity(1, "km"), u.Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(u.Quantity(1.0, "km"), u.Quantity(0, "deg"), u.Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(u.Quantity(1, "km"), u.Quantity(0, "deg"), u.Quantity(0, "km"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    return convert(convert(obj, u.AbstractQuantity), Distance)


# ===================================================================
# Unxt


@conversion_method(type_from=AbstractPos, type_to=u.AbstractQuantity)  # type: ignore[arg-type,type-abstract]
def convert_pos_to_absquantity(obj: AbstractPos, /) -> u.AbstractQuantity:
    """`coordinax.AbstractPos` -> `unxt.AbstractQuantity`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> pos = cx.vecs.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1.0], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.vecs.PolarPos(u.Quantity(1, "km"), u.Quantity(0, "deg"))
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(u.Quantity(1.0, "km"), u.Quantity(0, "deg"), u.Quantity(0, "deg"))
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(u.Quantity(1, "km"), u.Quantity(0, "deg"), u.Quantity(0, "km"))
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    cart = full_shaped(obj.vconvert(obj.cartesian_type))
    return jnp.stack(tuple(field_values(cart)), axis=-1)


@conversion_method(type_from=AbstractPos, type_to=u.Quantity)  # type: ignore[arg-type,type-abstract]
def convert_pos_to_q(obj: AbstractPos, /) -> u.Quantity["length"]:
    """`coordinax.AbstractPos` -> `unxt.Quantity`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> pos = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1, 2], dtype=int32), unit='km')

    >>> pos = cx.vecs.PolarPos(u.Quantity(1, "km"), u.Quantity(0, "deg"))
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(u.Quantity(1.0, "km"), u.Quantity(0, "deg"), u.Quantity(0, "deg"))
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(u.Quantity(1, "km"), u.Quantity(0, "deg"), u.Quantity(0, "km"))
    >>> convert(pos, u.AbstractQuantity)
    Quantity(Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    return convert(convert(obj, u.AbstractQuantity), u.Quantity)


@conversion_method(type_from=AbstractPos, type_to=BareQuantity)  # type: ignore[arg-type,type-abstract]
def convert_pos_to_uncheckedq(
    obj: AbstractPos, /
) -> Shaped[BareQuantity, "*batch dims"]:
    """`coordinax.AbstractPos` -> `unxt.BareQuantity`.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt.quantity import BareQuantity
    >>> import coordinax as cx

    >>> pos = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> convert(pos, BareQuantity)
    BareQuantity(Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1], "km")
    >>> convert(pos, BareQuantity)
    BareQuantity(Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> convert(pos, BareQuantity)
    BareQuantity(Array([1, 2], dtype=int32), unit='km')

    >>> pos = cx.vecs.PolarPos(u.Quantity(1, "km"), u.Quantity(0, "deg"))
    >>> convert(pos, BareQuantity)
    BareQuantity(Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, BareQuantity)
    BareQuantity(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(u.Quantity(1.0, "km"), u.Quantity(0, "deg"), u.Quantity(0, "deg"))
    >>> convert(pos, BareQuantity)
    BareQuantity(Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(u.Quantity(1, "km"), u.Quantity(0, "deg"), u.Quantity(0, "km"))
    >>> convert(pos, BareQuantity)
    BareQuantity(Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    return convert(convert(obj, u.AbstractQuantity), BareQuantity)
