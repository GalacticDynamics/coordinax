"""Register `plum.convert` to/from distances."""

__all__: list[str] = []

from plum import conversion_method

import unxt as u

from .measures import Distance, DistanceModulus, Parallax


@conversion_method(type_from=u.AbstractQuantity, type_to=Distance)  # type: ignore[arg-type]
def convert_quantity_to_distance(q: u.AbstractQuantity, /) -> Distance:
    """Convert any quantity to a Distance.

    Examples
    --------
    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity
    >>> from coordinax.distance import Distance
    >>> q = BareQuantity(1, "m")
    >>> q
    BareQuantity(Array(1, dtype=int32, ...), unit='m')

    >>> convert(q, Distance)
    Distance(Array(1, dtype=int32, ...), unit='m')

    The self-conversion doesn't copy the object:

    >>> q = Distance(1, "m")
    >>> convert(q, Distance) is q
    True

    """
    return q if isinstance(q, Distance) else Distance.from_(q)


@conversion_method(type_from=u.AbstractQuantity, type_to=DistanceModulus)  # type: ignore[arg-type]
def convert_quantity_to_distmod(q: u.AbstractQuantity, /) -> DistanceModulus:
    """Convert any quantity to a DistanceModulus.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> from coordinax.distance import DistanceModulus
    >>> q = u.Quantity(1, "mag")
    >>> q
    Quantity(Array(1, dtype=int32, ...), unit='mag')

    >>> convert(q, DistanceModulus)
    DistanceModulus(Array(1, dtype=int32, ...), unit='mag')

    The self-conversion doesn't copy the object:

    >>> q = DistanceModulus(1, "mag")
    >>> convert(q, DistanceModulus) is q
    True

    """
    return q if isinstance(q, DistanceModulus) else DistanceModulus.from_(q)


@conversion_method(type_from=u.AbstractQuantity, type_to=Parallax)  # type: ignore[arg-type]
def convert_quantity_to_parallax(q: u.AbstractQuantity, /) -> Parallax:
    """Convert any quantity to a Parallax.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> from coordinax.distance import Parallax
    >>> q = u.Quantity(1, "mas")
    >>> q
    Quantity(Array(1, dtype=int32, ...), unit='mas')

    >>> convert(q, Parallax)
    Parallax(Array(1, dtype=int32, ...), unit='mas')

    The self-conversion doesn't copy the object:

    >>> q = Parallax(1, "mas")
    >>> convert(q, Parallax) is q
    True

    """
    return q if isinstance(q, Parallax) else Parallax.from_(q)
