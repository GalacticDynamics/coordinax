"""Base classes for angular quantities."""

__all__: list[str] = []


import unxt as u
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
    Angle must have dimensions angle.

    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != angle_dimension:
            msg = "Angle must have dimensions angle."
            raise ValueError(msg)
