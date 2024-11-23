"""Compatibility for Quantity."""

__all__: list[str] = []

from plum import add_promotion_rule, conversion_method

from unxt.quantity import AbstractQuantity, Quantity

from .base import AbstractAngle
from .core import Angle

# Add a rule that when a AbstractAngle interacts with a Quantity, the
# angle degrades to a Quantity. This is necessary for many operations, e.g.
# division of an angle by non-dimensionless quantity where the resulting units
# are not those of an angle.
add_promotion_rule(AbstractAngle, Quantity, Quantity)


@conversion_method(type_from=AbstractAngle, type_to=Quantity)  # type: ignore[misc]
def _convert_angle_to_quantity(x: AbstractAngle) -> Quantity:
    """Convert a distance to a quantity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax.angle import Angle
    >>> from plum import convert

    >>> a = Angle(90, "deg")
    >>> convert(a, Quantity)
    Quantity['angle'](Array(90, dtype=int32, weak_type=True), unit='deg')

    """
    return Quantity(x.value, x.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Angle)  # type: ignore[misc]
def _convert_quantity_to_angle(q: AbstractQuantity, /) -> Angle:
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
