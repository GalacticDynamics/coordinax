"""Compatibility for Quantity."""

__all__: list[str] = []

from plum import conversion_method

from unxt import AbstractQuantity

from .core import Distance, DistanceModulus, Parallax

#####################################################################
# Conversion


@conversion_method(type_from=AbstractQuantity, type_to=Distance)  # type: ignore[misc]
def _quantity_to_distance(q: AbstractQuantity, /) -> Distance:
    """Convert any quantity to a Distance.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt import Quantity, UncheckedQuantity
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
    >>> from unxt import Quantity
    >>> from coordinax.distance import Parallax
    >>> q = Quantity(1, "mas")
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
    >>> from unxt import Quantity
    >>> from coordinax.distance import DistanceModulus
    >>> q = Quantity(1, "mag")
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
