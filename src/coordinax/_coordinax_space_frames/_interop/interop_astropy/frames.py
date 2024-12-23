"""Coordinax interop with Astropy package."""

__all__: list[str] = []

import astropy.coordinates as apyc
import equinox as eqx

from coordinax._coordinax_space_frames import ICRS, Galactocentric
from coordinax._src.frames.base import AbstractReferenceFrame
from coordinax._src.vectors.d3 import LonLatSphericalPos


@AbstractReferenceFrame.from_.dispatch
def from_(cls: type[ICRS], obj: apyc.ICRS, /) -> ICRS:
    """Construct from a `astropy.coordinates.ICRS`.

    Examples
    --------
    >>> import astropy.coordinates as apyc
    >>> import coordinax.frames as cxf

    >>> apy_icrs = apyc.ICRS()
    >>> cxf.ICRS.from_(apy_icrs)
    ICRS()

    """
    obj = eqx.error_if(obj, obj.has_data, "Astropy frame must not have data.")
    return cls()


@AbstractReferenceFrame.from_.dispatch
def from_(cls: type[Galactocentric], obj: apyc.Galactocentric, /) -> Galactocentric:
    """Construct from a `astropy.coordinates.Galactocentric`.

    Examples
    --------
    >>> import astropy.coordinates as apyc
    >>> import coordinax.frames as cxf

    >>> apy_gcf = apyc.Galactocentric()
    >>> apy_gcf
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
    (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

    >>> gcf = cxf.Galactocentric.from_(apy_gcf)
    >>> gcf
    Galactocentric(
        galcen=LonLatSphericalPos( ... ),
        roll=Quantity[...](value=f32[], unit=Unit("deg")),
        z_sun=Quantity[...](value=f32[], unit=Unit("pc")),
        galcen_v_sun=CartesianVel3D( ... )
    )

    Checking equality

    >>> (gcf.galcen.lon.ustrip("deg") == apy_gcf.galcen_coord.ra.to_value("deg")
    ...  and gcf.galcen.lat.ustrip("deg") == apy_gcf.galcen_coord.dec.to_value("deg")
    ...  and gcf.galcen.distance.ustrip("kpc") == apy_gcf.galcen_distance.to_value("kpc") )
    Array(True, dtype=bool)


    """  # noqa: E501
    obj = eqx.error_if(obj, obj.has_data, "Astropy frame must not have data.")
    galcen = LonLatSphericalPos(
        lon=obj.galcen_coord.ra,
        lat=obj.galcen_coord.dec,
        distance=obj.galcen_distance,
    )
    return cls(galcen, roll=obj.roll, z_sun=obj.z_sun, galcen_v_sun=obj.galcen_v_sun)
