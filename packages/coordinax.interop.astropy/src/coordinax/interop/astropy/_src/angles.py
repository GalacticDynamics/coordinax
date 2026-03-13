"""Astropy Angle compatibility."""

__all__: tuple[str, ...] = ()


from plum import conversion_method

from astropy.coordinates import Angle as AstropyAngle

import coordinax.angles as cxa


# TODO: move this to unxt
@conversion_method(type_from=AstropyAngle, type_to=cxa.Angle)  # type: ignore[arg-type]
def convert_astropy_angle_to_coordinax_angle(q: AstropyAngle, /) -> cxa.Angle:
    """Convert a `astropy.coordinates.Angle` to a `coordinax.angles.Angle`.

    Examples
    --------
    >>> import astropy.coordinates as apyc
    >>> import plum
    >>> import coordinax.angles as cxa

    >>> plum.convert(apyc.Angle(1.0, "rad"), cxa.Angle)
    Angle(1., 'rad')

    """
    return cxa.Angle(q.value, q.unit)
