"""Register `plum.convert` to/from distances."""

__all__: list[str] = []

from plum import conversion_method

import unxt as u
from unxt.quantity import AbstractQuantity

from .distance import Distance, DistanceModulus


@conversion_method(type_from=AbstractQuantity, type_to=Distance)  # type: ignore[misc]
def convert_quantity_to_distance(q: AbstractQuantity, /) -> Distance:
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

    unit = u.unit_of(q)
    return Distance(q.ustrip(unit), unit)


@conversion_method(type_from=AbstractQuantity, type_to=DistanceModulus)  # type: ignore[misc]
def convert_quantity_to_distmod(q: AbstractQuantity, /) -> DistanceModulus:
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

    unit = u.unit_of(q)
    return DistanceModulus(q.ustrip(unit), unit)
