"""Astropy Quantity compatibility."""

__all__: tuple[str, ...] = ()


from plum import conversion_method

import astropy.units as apyu

import coordinax.astro as cxastro
import coordinax.distances as cxd

# ============================================================================
# Distance


@conversion_method(type_from=apyu.Quantity, type_to=cxd.Distance)
def convert_astropy_quantity_to_unxt_distance(q: apyu.Quantity, /) -> cxd.Distance:
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


@conversion_method(type_from=apyu.Quantity, type_to=cxastro.Parallax)
def convert_astropy_quantity_to_unxt_parallax(q: apyu.Quantity, /) -> cxastro.Parallax:
    """Convert a `astropy.units.Quantity` to a `coordinax.astro.Parallax`.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import plum
    >>> import coordinax.astro as cxastro

    >>> plum.convert(apyu.Quantity(1.0, "radian"), cxastro.Parallax)
    Parallax(1., 'rad')

    """
    return cxastro.Parallax(q.value, q.unit)


# ============================================================================
# Distance Modulus


@conversion_method(type_from=apyu.Quantity, type_to=cxastro.DistanceModulus)
def convert_astropy_quantity_to_unxt_distmod(
    q: apyu.Quantity, /
) -> cxastro.DistanceModulus:
    """Convert a `astropy.units.Quantity` to a `coordinax.astro.DistanceModulus`.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import plum
    >>> import coordinax.astro as cxastro

    >>> plum.convert(apyu.Quantity(1.0, "mag"), cxastro.DistanceModulus)
    DistanceModulus(1., 'mag')

    """
    return cxastro.DistanceModulus(q.value, q.unit)
