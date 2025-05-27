"""Base classes for angular quantities."""

__all__: list[str] = [
    "AbstractAngle",
    "wrap_to",
]


from dataclasses import replace

import equinox as eqx
from jaxtyping import Array, Shaped
from plum import add_promotion_rule

import unxt as u
from unxt._src.units.api import AstropyUnits

angle_dimension = u.dimension("angle")


class AbstractAngle(u.AbstractQuantity):  # type: ignore[misc]
    """Angular Quantity.

    See Also
    --------
    `coordinax.angle.Angle` : a concrete implementation of this class.

    Examples
    --------
    For this example, we will use the concrete implementation of
    `coordinax.angle.AbstractAngle`, `coordinax.angle.Angle`.

    >>> from coordinax.angle import Angle

    >>> Angle(90, "deg")
    Angle(Array(90, dtype=int32, ...), unit='deg')

    Angles have to have dimensions of angle.

    >>> try: Angle(90, "m")
    ... except ValueError as e: print(e)
    Angle must have units with angular dimensions.

    """

    value: eqx.AbstractVar[Shaped[Array, "*shape"]]
    """The value of the `unxt.AbstractQuantity`."""

    unit: eqx.AbstractVar[AstropyUnits]
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != angle_dimension:
            msg = f"{type(self).__name__} must have units with angular dimensions."
            raise ValueError(msg)

    def wrap_to(
        self, /, min: u.AbstractQuantity, max: u.AbstractQuantity
    ) -> "AbstractAngle":
        """Wrap the angle to the range [min, max).

        Parameters
        ----------
        min, max
            The minimum, maximum value of the range.

        Examples
        --------
        >>> import unxt as u
        >>> from coordinax.angle import Angle

        >>> angle = Angle(370, "deg")
        >>> angle.wrap_to(min=u.Quantity(0, "deg"), max=u.Quantity(360, "deg"))
        Angle(Array(10, dtype=int32, ...), unit='deg')

        """
        return wrap_to(self, min=min, max=max)


# Add a rule that when a AbstractAngle interacts with a Quantity, the
# angle degrades to a Quantity. This is necessary for many operations, e.g.
# division of an angle by non-dimensionless quantity where the resulting units
# are not those of an angle.
add_promotion_rule(AbstractAngle, u.Quantity, u.Quantity)
add_promotion_rule(AbstractAngle, u.quantity.BareQuantity, u.quantity.BareQuantity)


# =========================================================


def wrap_to(
    angle: AbstractAngle,
    /,
    min: u.AbstractQuantity,
    max: u.AbstractQuantity,
) -> AbstractAngle:
    """Wrap the angle to the range [min, max).

    Parameters
    ----------
    angle
        The angle to wrap.
    min, max
        The minimum, maximum value of the range.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.angle import Angle, wrap_to

    >>> angle = Angle(370, "deg")
    >>> wrap_to(angle, min=u.Quantity(0, "deg"), max=u.Quantity(360, "deg"))
    Angle(Array(10, dtype=int32, ...), unit='deg')

    """
    minv, maxv = min.ustrip(angle.unit), max.ustrip(angle.unit)
    value = ((angle.value - minv) % (maxv - minv)) + minv
    return replace(angle, value=value)
