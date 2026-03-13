"""Astropy Quantity compatibility."""

__all__: tuple[str, ...] = ()


from plum import conversion_method

from astropy.units import Quantity as AstropyQuantity

import coordinax.distances as cxd

# ============================================================================
# Distance


@conversion_method(type_from=AstropyQuantity, type_to=cxd.Distance)  # type: ignore[arg-type]
def convert_astropy_quantity_to_unxt_distance(q: AstropyQuantity, /) -> cxd.Distance:
    """Convert a `astropy.units.Quantity` to a `coordinax.distances.Distance`.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import plum
    >>> import coordinax.distances as cxd

    >>> plum.convert(apyu.Quantity(1.0, "cm"), cxd.Distance)
    Distance(1., 'cm')

    """
    return cxd.Distance(q.value, q.unit)


# ============================================================================
# Parallax


@conversion_method(type_from=AstropyQuantity, type_to=cxd.Parallax)  # type: ignore[arg-type]
def convert_astropy_quantity_to_unxt_parallax(q: AstropyQuantity, /) -> cxd.Parallax:
    """Convert a `astropy.units.Quantity` to a `coordinax.distance.Parallax`.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import plum
    >>> import coordinax.distances as cxd

    >>> plum.convert(apyu.Quantity(1.0, "radian"), cxd.Parallax)
    Parallax(1., 'rad')

    """
    return cxd.Parallax(q.value, q.unit)


# ============================================================================
# Distance Modulus


@conversion_method(type_from=AstropyQuantity, type_to=cxd.DistanceModulus)  # type: ignore[arg-type]
def convert_astropy_quantity_to_unxt_distmod(
    q: AstropyQuantity, /
) -> cxd.DistanceModulus:
    """Convert a `astropy.units.Quantity` to a `coordinax.distances.DistanceModulus`.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import plum
    >>> import coordinax.distances as cxd

    >>> plum.convert(apyu.Quantity(1.0, "mag"), cxd.DistanceModulus)
    DistanceModulus(1., 'mag')

    """
    return cxd.DistanceModulus(q.value, q.unit)
