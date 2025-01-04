"""Compatibility for Angles."""

__all__: list[str] = []

from plum import conversion_method

import unxt as u
from unxt.quantity import AbstractQuantity

from .angle import Angle
from .parallax import Parallax


@conversion_method(type_from=AbstractQuantity, type_to=Angle)  # type: ignore[misc]
def convert_quantity_to_angle(q: AbstractQuantity, /) -> Angle:
    """Convert any quantity to an Angle.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity, UncheckedQuantity
    >>> from coordinax.angle import Angle
    >>> q = UncheckedQuantity(1, "rad")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='rad')

    >>> convert(q, Angle)
    Angle(Array(1, dtype=int32, ...), unit='rad')

    The self-conversion doesn't copy the object:

    >>> q = Angle(1, "rad")
    >>> convert(q, Angle) is q
    True

    """
    if isinstance(q, Angle):
        return q

    unit = u.unit_of(q)
    return Angle(q.ustrip(unit), unit)


@conversion_method(type_from=AbstractQuantity, type_to=Parallax)  # type: ignore[misc]
def convert_quantity_to_parallax(q: AbstractQuantity, /) -> Parallax:
    """Convert any quantity to a Parallax.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> from coordinax.distance import Parallax
    >>> q = u.Quantity(1, "mas")
    >>> q
    Quantity['angle'](Array(1, dtype=int32, ...), unit='mas')

    >>> convert(q, Parallax)
    Parallax(Array(1, dtype=int32, weak_type=True), unit='mas')

    The self-conversion doesn't copy the object:

    >>> q = Parallax(1, "mas")
    >>> convert(q, Parallax) is q
    True

    """
    if isinstance(q, Parallax):
        return q

    unit = u.unit_of(q)
    return Parallax(q.ustrip(unit), unit)
