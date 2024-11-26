"""Compatibility for Quantity."""

__all__: list[str] = []

from plum import add_promotion_rule, conversion_method

from unxt.quantity import AbstractQuantity, Quantity

from .base import AbstractDistance
from .core import Distance, DistanceModulus, Parallax

# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, Quantity, Quantity)

#####################################################################
# Conversion


@conversion_method(type_from=AbstractDistance, type_to=Quantity)  # type: ignore[misc]
def _convert_distance_to_quantity(x: AbstractDistance) -> Quantity:
    """Convert a distance to a quantity."""
    return Quantity(x.value, x.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Distance)  # type: ignore[misc]
def _quantity_to_distance(q: AbstractQuantity, /) -> Distance:
    """Convert any quantity to a Distance.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import Quantity, UncheckedQuantity
    >>> from coordinax.distance import Distance
    >>> q = UncheckedQuantity(1, "m")
    >>> q
    UncheckedQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, Distance)
    Distance(Array(1, dtype=int32, ...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = Distance(1, "m")
    >>> convert(q, Distance) is q
    True

    """
    if isinstance(q, Distance):
        return q
    return Distance(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=Parallax)  # type: ignore[misc]
def _quantity_to_parallax(q: AbstractQuantity, /) -> Parallax:
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
    return Parallax(q.value, q.unit)


@conversion_method(type_from=AbstractQuantity, type_to=DistanceModulus)  # type: ignore[misc]
def _quantity_to_distmod(q: AbstractQuantity, /) -> DistanceModulus:
    """Convert any quantity to a DistanceModulus.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> from coordinax.distance import DistanceModulus
    >>> q = u.Quantity(1, "mag")
    >>> q
    Quantity['dex'](Array(1, dtype=int32, ...), unit='mag')

    >>> convert(q, DistanceModulus)
    DistanceModulus(Array(1, dtype=int32, ...), unit='mag')

    The self-conversion doesn't copy the object:

    >>> q = DistanceModulus(1, "mag")
    >>> convert(q, DistanceModulus) is q
    True

    """
    if isinstance(q, DistanceModulus):
        return q
    return DistanceModulus(q.value, q.unit)
