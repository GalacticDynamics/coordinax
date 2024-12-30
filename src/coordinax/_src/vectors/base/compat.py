"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

from unxt.quantity import AbstractQuantity

from .base_pos import AbstractPos
from coordinax._src.distances import Distance


@conversion_method(type_from=AbstractPos, type_to=Distance)  # type: ignore[misc]
def convert_pos_to_distance(obj: AbstractPos, /) -> Shaped[Distance, "*batch dims"]:
    """`coordinax.AbstractPos` -> `coordinax.Distance`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity
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

    >>> pos = cx.vecs.PolarPos(Quantity(1, "km"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(Quantity(1, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Distance)
