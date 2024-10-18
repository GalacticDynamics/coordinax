"""Astropy Quantity compatibility."""

__all__: list[str] = []


from astropy.units import Quantity as AstropyQuantity
from plum import conversion_method

from coordinax.d import Distance, DistanceModulus, Parallax

# ============================================================================
# Distance


@conversion_method(type_from=AstropyQuantity, type_to=Distance)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_distance(q: AstropyQuantity, /) -> Distance:
    """Convert a `astropy.units.Quantity` to a `coordinax.d.Distance`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from coordinax.d import Distance

    >>> convert(AstropyQuantity(1.0, "cm"), Distance)
    Distance(Array(1., dtype=float32), unit='cm')

    """
    return Distance(q.value, q.unit)


# ============================================================================
# Parallax


@conversion_method(type_from=AstropyQuantity, type_to=Parallax)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_parallax(q: AstropyQuantity, /) -> Parallax:
    """Convert a `astropy.units.Quantity` to a `coordinax.d.Parallax`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from coordinax.d import Parallax

    >>> convert(AstropyQuantity(1.0, "radian"), Parallax)
    Parallax(Array(1., dtype=float32), unit='rad')

    """
    return Parallax(q.value, q.unit)


# ============================================================================
# Distance Modulus


@conversion_method(type_from=AstropyQuantity, type_to=DistanceModulus)  # type: ignore[misc]
def convert_astropy_quantity_to_unxt_distmod(q: AstropyQuantity, /) -> DistanceModulus:
    """Convert a `astropy.units.Quantity` to a `coordinax.d.DistanceModulus`.

    Examples
    --------
    >>> from astropy.units import Quantity as AstropyQuantity
    >>> from plum import convert
    >>> from coordinax.d import DistanceModulus

    >>> convert(AstropyQuantity(1.0, "mag"), DistanceModulus)
    DistanceModulus(Array(1., dtype=float32), unit='mag')

    """
    return DistanceModulus(q.value, q.unit)
