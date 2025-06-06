"""Compatibility for Angles."""

__all__: list[str] = []

from plum import conversion_method

import unxt as u

from .angle import Angle


@conversion_method(type_from=u.AbstractQuantity, type_to=Angle)  # type: ignore[arg-type]
def convert_quantity_to_angle(q: u.AbstractQuantity, /) -> Angle:
    """Convert any quantity to an Angle.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity
    >>> from coordinax.angle import Angle
    >>> q = BareQuantity(1, "rad")
    >>> q
    BareQuantity(Array(1, dtype=int32, ...), unit='rad')

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
