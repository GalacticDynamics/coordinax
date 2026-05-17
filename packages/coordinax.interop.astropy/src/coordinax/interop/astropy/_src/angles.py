"""Astropy Angle compatibility."""

__all__: tuple[str, ...] = ("convert_astropy_angle_to_cx_angle",)


import plum

from astropy.coordinates import Angle as AstropyAngle

import coordinax.angles as cxa


# TODO: move this to unxt
@plum.conversion_method(type_from=AstropyAngle, type_to=cxa.Angle)
def convert_astropy_angle_to_cx_angle(q: AstropyAngle, /) -> cxa.Angle:
    """Convert a `astropy.coordinates.Angle` to a `coordinax.angles.Angle`.

    >>> import astropy.coordinates as apyc
    >>> import plum
    >>> import coordinax.angles as cxa
    >>> plum.convert(apyc.Angle(1.0, "rad"), cxa.Angle)
    Angle(1., 'rad')

    """
    return cxa.Angle(q.value, q.unit)
