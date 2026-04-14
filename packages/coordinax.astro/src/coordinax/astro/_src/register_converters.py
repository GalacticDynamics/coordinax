"""Register `plum.convert` to/from distances."""

__all__: tuple[str, ...] = ()

import plum

import unxt as u

from .distance_modulus import DistanceModulus
from .parallax import Parallax


@plum.conversion_method(type_from=u.AbstractQuantity, type_to=DistanceModulus)
def convert_quantity_to_distmod(q: u.AbstractQuantity, /) -> DistanceModulus:
    """Convert any quantity to a DistanceModulus.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> from coordinax.astro import DistanceModulus
    >>> q = u.Q(1, "mag")
    >>> q
    Q(1, 'mag')

    >>> convert(q, DistanceModulus)
    DistanceModulus(1, 'mag')

    The self-conversion doesn't copy the object:

    >>> q = DistanceModulus(1, "mag")
    >>> convert(q, DistanceModulus) is q
    True

    """
    return q if isinstance(q, DistanceModulus) else DistanceModulus.from_(q)  # ty: ignore[invalid-return-type]


@plum.conversion_method(type_from=u.AbstractQuantity, type_to=Parallax)
def convert_quantity_to_parallax(q: u.AbstractQuantity, /) -> Parallax:
    """Convert any quantity to a Parallax.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> from coordinax.astro import Parallax
    >>> q = u.Q(1, "mas")
    >>> q
    Q(1, 'mas')

    >>> convert(q, Parallax)
    Parallax(1, 'mas')

    The self-conversion doesn't copy the object:

    >>> q = Parallax(1, "mas")
    >>> convert(q, Parallax) is q
    True

    """
    return q if isinstance(q, Parallax) else Parallax.from_(q)  # ty: ignore[invalid-return-type]
