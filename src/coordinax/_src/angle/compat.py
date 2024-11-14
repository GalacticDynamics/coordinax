"""Compatibility for Quantity."""

__all__: list[str] = []

from plum import conversion_method

from unxt.quantity import AbstractQuantity

from .core import Angle


@conversion_method(type_from=AbstractQuantity, type_to=Angle)  # type: ignore[misc]
def _quantity_to_angle(q: AbstractQuantity, /) -> Angle:
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
    return Angle(q.value, q.unit)
