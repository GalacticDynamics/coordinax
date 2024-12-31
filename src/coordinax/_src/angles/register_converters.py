"""Compatibility for Angles."""

__all__: list[str] = []

from plum import conversion_method

import unxt as u
from unxt.quantity import AbstractQuantity, UncheckedQuantity

from .angle import Angle
from .base import AbstractAngle


@conversion_method(type_from=AbstractAngle, type_to=u.Quantity)  # type: ignore[misc]
def convert_angle_to_quantity(x: AbstractAngle) -> u.Quantity:
    """Convert a distance to a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.angle import Angle
    >>> from plum import convert

    >>> a = Angle(90, "deg")
    >>> convert(a, u.Quantity)
    Quantity['angle'](Array(90, dtype=int32, weak_type=True), unit='deg')

    """
    unit = u.unit_of(x)
    return u.Quantity(x.ustrip(unit), unit)


@conversion_method(type_from=AbstractAngle, type_to=UncheckedQuantity)  # type: ignore[misc]
def convert_angle_to_uncheckedquantity(x: AbstractAngle) -> UncheckedQuantity:
    """Convert a distance to a quantity.

    Examples
    --------
    >>> from unxt.quantity import UncheckedQuantity
    >>> from coordinax.angle import Angle
    >>> from plum import convert

    >>> a = Angle(90, "deg")
    >>> convert(a, UncheckedQuantity)
    UncheckedQuantity(Array(90, dtype=int32, weak_type=True), unit='deg')

    """
    unit = u.unit_of(x)
    return UncheckedQuantity(x.ustrip(unit), unit)


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
