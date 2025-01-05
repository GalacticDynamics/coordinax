"""Base classes for angular quantities."""

__all__: list[str] = []


import equinox as eqx
from jaxtyping import Array, Shaped
from plum import add_promotion_rule

import unxt as u
from unxt._src.units.api import AbstractUnits
from unxt.quantity import AbstractQuantity

angle_dimension = u.dimension("angle")


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
    Angle must have units with angular dimensions.

    """

    value: eqx.AbstractVar[Shaped[Array, "*shape"]]
    """The value of the `AbstractQuantity`."""

    unit: eqx.AbstractVar[AbstractUnits]
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != angle_dimension:
            msg = f"{type(self).__name__} must have units with angular dimensions."
            raise ValueError(msg)


# Add a rule that when a AbstractAngle interacts with a Quantity, the
# angle degrades to a Quantity. This is necessary for many operations, e.g.
# division of an angle by non-dimensionless quantity where the resulting units
# are not those of an angle.
add_promotion_rule(AbstractAngle, u.Quantity, u.Quantity)
