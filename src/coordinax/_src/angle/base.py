"""Base classes for angular quantities."""

__all__: list[str] = []

import astropy.units as u
from plum import add_promotion_rule, conversion_method

from unxt import Quantity, dimensions_of
from unxt.quantity import AbstractQuantity

angle_dimension = u.get_physical_type("angle")


class AbstractAngle(AbstractQuantity):  # type: ignore[misc]
    """Angular Quantity.

    See Also
    --------
    `coordinax.angle.Angle` : a concrete implementation of this class.

    Examples
    --------
    For this example, we will use the concrete implementation of
    `coordinax.angle.AbstractAngle`, `coordinax.angle.Angle`.

    >>> from coordinax.angle import Angle

    >>> a = Angle(90, "deg")
    >>> a
    Angle(Array(90, dtype=int32, weak_type=True), unit='deg')

    Angles have to have dimensions of angle.

    >>> try: Angle(90, "m")
    ... except ValueError as e: print(e)
    Angle must have dimensions angle.

    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if dimensions_of(self) != angle_dimension:
            msg = "Angle must have dimensions angle."
            raise ValueError(msg)


# ============================================================================
# Conversion and Promotion

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
