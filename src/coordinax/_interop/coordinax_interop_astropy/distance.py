"""Astropy Quantity compatibility."""

__all__: list[str] = []


from astropy.units import Quantity as AstropyQuantity
from plum import conversion_method

from coordinax.distance import Distance, DistanceModulus, Parallax

# ============================================================================
# Distance


@conversion_method(type_from=AstropyQuantity, type_to=Distance)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_distance(q: AstropyQuantity, /) -> Distance:
    """Convert a `astropy.units.Quantity` to a `coordinax.Distance`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> import coordinax as cx

    >>> convert(AstropyQuantity(1.0, "cm"), cx.Distance)
    Distance(Array(1., dtype=float32), unit='cm')

    """
    return Distance(q.value, q.unit)


# ============================================================================
# Parallax


@conversion_method(type_from=AstropyQuantity, type_to=Parallax)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_parallax(q: AstropyQuantity, /) -> Parallax:
    """Convert a `astropy.units.Quantity` to a `coordinax.distance.Parallax`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from coordinax.distance import Parallax

    >>> convert(AstropyQuantity(1.0, "radian"), Parallax)
    Parallax(Array(1., dtype=float32), unit='rad')

    """
    return Parallax(q.value, q.unit)


# ============================================================================
# Distance Modulus


@conversion_method(type_from=AstropyQuantity, type_to=DistanceModulus)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_distmod(q: AstropyQuantity, /) -> DistanceModulus:
    """Convert a `astropy.units.Quantity` to a `coordinax.distance.DistanceModulus`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from coordinax.distance import DistanceModulus

    >>> convert(AstropyQuantity(1.0, "mag"), DistanceModulus)
    DistanceModulus(Array(1., dtype=float32), unit='mag')

    """
    return DistanceModulus(q.value, q.unit)
