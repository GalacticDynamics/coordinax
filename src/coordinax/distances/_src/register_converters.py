"""Register `plum.convert` to/from distances."""

__all__: tuple[str, ...] = ()

from typing import cast

from plum import conversion_method

import unxt as u

from .measures import Distance


@conversion_method(type_from=u.AbstractQuantity, type_to=Distance)
def convert_quantity_to_distance(q: u.AbstractQuantity, /) -> Distance:
    """Convert any quantity to a Distance.

    >>> from plum import convert
    >>> from unxt.quantity import BareQuantity
    >>> from coordinax.distances import Distance
    >>> q = BareQuantity(1, "m")
    >>> q
    BareQuantity(1, 'm')

    >>> convert(q, Distance)
    Distance(1, 'm')

    The self-conversion doesn't copy the object:

    >>> q = Distance(1, "m")
    >>> convert(q, Distance) is q
    True

    """
    out = q if isinstance(q, Distance) else Distance.from_(q)
    return cast("Distance", out)
