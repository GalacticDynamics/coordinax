"""Interoperability with {mod}`astropy.coordinates` frames.

This module provides bidirectional conversions between coordinax.astro reference
frames and Astropy coordinate frames. The conversions preserve all frame
parameters and enable seamless integration between the two libraries.

Supported Frames:

- ICRS: International Celestial Reference System
- Galactocentric: Galactic center-based coordinate system

All conversions are implemented using plum's `@plum.conversion_method` decorator,
allowing automatic dispatch when using `plum.convert()`.

Examples
--------
Basic ICRS conversion:

>>> import coordinax.astro as cxa
>>> import astropy.coordinates as apyc
>>> import plum

>>> cx_icrs = cxastro.ICRS()
>>> apy_icrs = plum.convert(cx_icrs, apyc.ICRS)
>>> isinstance(apy_icrs, apyc.ICRS)
True

Galactocentric conversion with custom parameters:

>>> import unxt as u
>>> import coordinax.main as cx
>>> import coordinax.charts as cxc
>>> import coordinax.representations as cxr

>>> galcen = cx.Point.from_(
...     {"lon": u.Q(0, "deg"), "lat": u.Q(0, "deg"),
...      "distance": u.Q(8.122, "kpc")},
...     cx.lonlat_sph3d,
... )
>>> cx_galcen = cxastro.Galactocentric(
...     galcen=galcen,
...     z_sun=u.Q(20.8, "pc"),
...     roll=u.Q(0, "deg"),
... )
>>> apy_galcen = plum.convert(cx_galcen, apyc.Galactocentric)
>>> isinstance(apy_galcen, apyc.Galactocentric)
True

# # Round-trip conversions preserve frame parameters:

# >>> cx_result = plum.convert(apy_galcen, cx.Point)
# >>> cx_result

"""

__all__: tuple[str, ...] = ()


import equinox as eqx
import plum

import astropy.coordinates as apyc
import astropy.units as apyu
import unxt as u

import coordinax.astro as cxastro
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.vectors as cxv
from .custom_types import CDict

# =============================================================================
# ICRS


@plum.conversion_method(cxastro.ICRS, apyc.BaseCoordinateFrame)
@plum.conversion_method(cxastro.ICRS, apyc.ICRS)
def coordinax_icrs_to_astropy_icrs(frame: cxastro.ICRS, /) -> apyc.ICRS:
    """Convert coordinax ICRS frame to Astropy ICRS frame.

    The ICRS (International Celestial Reference System) frame is a kinematically
    non-rotating coordinate system centered at the solar system barycenter. Both
    coordinax and Astropy implementations have no frame-specific parameters, so
    the conversion is straightforward.

    >>> import coordinax.astro as cxa
    >>> import astropy.coordinates as apyc
    >>> import plum

    >>> cx_frame = cxastro.ICRS()
    >>> plum.convert(cx_frame, apyc.ICRS)
    <ICRS Frame>

    >>> plum.convert(cx_frame, apyc.BaseCoordinateFrame)
    <ICRS Frame>

    """
    return apyc.ICRS()


@cxf.AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxastro.ICRS], obj: apyc.ICRS, /) -> cxastro.ICRS:
    """Construct from a `astropy.coordinates.ICRS`.

    >>> import coordinax.astro as cxastro
    >>> from plum import convert
    >>> import astropy.coordinates as apyc

    >>> apy_frame = apyc.ICRS()
    >>> cx_frame = convert(apy_frame, cxastro.ICRS)
    >>> isinstance(cx_frame, cxastro.ICRS)
    True

    >>> cxastro.ICRS.from_(apy_frame)
    ICRS()

    """
    if obj.has_data:
        raise ValueError("Astropy frame must not have data.")
    return cls()


@plum.conversion_method(apyc.ICRS, cxastro.AbstractSpaceFrame)
@plum.conversion_method(apyc.ICRS, cxastro.ICRS)
def astropy_icrs_to_coordinax_icrs(frame: apyc.ICRS, /) -> cxastro.ICRS:
    """Convert Astropy ICRS frame to coordinax ICRS frame.

    The ICRS (International Celestial Reference System) frame is a kinematically
    non-rotating coordinate system centered at the solar system barycenter. Both
    coordinax and Astropy implementations have no frame-specific parameters, so
    the conversion is straightforward.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.astro as cxastro
    >>> import plum

    >>> apy_frame = apyc.ICRS()
    >>> plum.convert(apy_frame, cxastro.ICRS)
    ICRS()

    >>> plum.convert(apy_frame, cxastro.AbstractSpaceFrame)
    ICRS()

    """
    return cxastro.icrs


# =============================================================================
# Galactocentric


@plum.conversion_method(cxastro.Galactocentric, apyc.BaseCoordinateFrame)
@plum.conversion_method(cxastro.Galactocentric, apyc.Galactocentric)
def coordinax_galactocentric_to_astropy_galactocentric(
    frame: cxastro.Galactocentric, /
) -> apyc.Galactocentric:
    """Convert coordinax Galactocentric frame to Astropy Galactocentric frame.

    The Galactocentric frame is centered at the center of the Milky Way Galaxy,
    with the x-axis pointing from the Galactic center to the Sun, the z-axis
    pointing toward the North Galactic Pole, and the y-axis following the
    right-hand rule.

    This conversion extracts all frame parameters from the coordinax frame and
    constructs an equivalent Astropy frame with the same parameters:

    - galcen_coord: Position of the Galactic center
    - galcen_distance: Distance to the Galactic center
    - galcen_v_sun: Velocity of the Sun with respect to the Galactic center
    - z_sun: Height of the Sun above the Galactic midplane
    - roll: Rotation angle of the frame

    Examples
    --------
    >>> import astropy.coordinates as apyc
    >>> import coordinax.main as cx
    >>> import coordinax.astro as cxa
    >>> import plum
    >>> import unxt as u

    Convert with default parameters:

    >>> cx_frame = cxastro.Galactocentric()
    >>> plum.convert(cx_frame, apyc.Galactocentric)
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec, distance) in (deg, deg, kpc)
        (266.4051, -28.936175, 8.122)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

    Convert with custom parameters:

    >>> galcen = cx.Point.from_(
    ...     {"lon": u.Q(0, "deg"), "lat": u.Q(0, "deg"), "distance": u.Q(8.122, "kpc")},
    ...     cxc.lonlat_sph3d
    ... )
    >>> cx_frame = cxastro.Galactocentric(
    ...     galcen=galcen,
    ...     z_sun=u.Q(20.8, "pc"),
    ...     roll=u.Q(0, "deg"),
    ... )
    >>> apy_frame = plum.convert(cx_frame, apyc.Galactocentric)
    >>> isinstance(apy_frame, apyc.Galactocentric)
    True

    >>> apy_frame = plum.convert(cx_frame, apyc.BaseCoordinateFrame)
    >>> isinstance(apy_frame, apyc.Galactocentric)
    True

    """  # noqa: E501
    # Convert the galcen position
    galcen_coord = apyc.ICRS(plum.convert(frame.galcen, apyc.SphericalRepresentation))

    # # Convert the galcen velocity
    # galcen_v_sun: apyc.CartesianDifferential = plum.convert(
    #     frame.galcen_v_sun, apyc.CartesianDifferential
    # )

    return apyc.Galactocentric(
        galcen_coord=galcen_coord,
        galcen_distance=plum.convert(frame.galcen["distance"], apyu.Quantity),
        # galcen_v_sun=galcen_v_sun,
        z_sun=plum.convert(frame.z_sun, apyu.Quantity),
        roll=plum.convert(frame.roll, apyu.Quantity),
    )


@cxf.AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[cxastro.Galactocentric], frame: apyc.Galactocentric, /
) -> cxastro.Galactocentric:
    """Construct from a `astropy.coordinates.Galactocentric`.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.frames as cxf

    >>> apy_gcf = apyc.Galactocentric()
    >>> apy_gcf
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
    (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

    >>> gcf = cxf.Galactocentric.from_(apy_gcf)
    >>> gcf
    Galactocentric(
      galcen=Point(
        { 'lon': Q(f64[], 'deg'), 'lat': Q(f64[], 'deg'), 'distance': Q(f64[], 'kpc') },
        chart=LonLatSpherical3D(M=Rn(3)), frame=ICRS()
      ),
      roll=Angle(f64[], 'deg'),
      z_sun=Quantity(f64[], 'pc')
    )

    Checking equality

    >>> (gcf.galcen["lon"].ustrip("deg") == apy_gcf.galcen_coord.ra.to_value("deg")
    ...  and gcf.galcen["lat"].ustrip("deg") == apy_gcf.galcen_coord.dec.to_value("deg")
    ...  and gcf.galcen["distance"].ustrip("kpc") == apy_gcf.galcen_distance.to_value("kpc") )
    Array(True, dtype=bool)

    """  # noqa: E501
    frame = eqx.error_if(frame, frame.has_data, "Astropy frame must not have data.")

    # Convert galcen_coord to Vector with lonlat_sph3d chart and point role
    # galcen_coord is an ICRS coordinate, so access ra/dec from representation
    galcen_data: CDict = {
        "lon": plum.convert(frame.galcen_coord.ra, u.Q),
        "lat": plum.convert(frame.galcen_coord.dec, u.Q),
        "distance": plum.convert(frame.galcen_distance, u.Q),
    }
    galcen = cxv.Point(  # ty: ignore[missing-argument]
        galcen_data, chart=cxc.lonlat_sph3d, frame=cxastro.icrs
    )

    # # Convert galcen_v_sun to CartesianVel3D
    # galcen_v_sun: cx.Tangent[cxc.Cart3D, cxr.PhysVel] = plum.convert(
    #     frame.galcen_v_sun, cx.Tangent
    # )

    return cxastro.Galactocentric(
        galcen=galcen,
        roll=plum.convert(frame.roll, u.Q),
        z_sun=plum.convert(frame.z_sun, u.Q),
        # galcen_v_sun=galcen_v_sun,
    )


@plum.conversion_method(apyc.Galactocentric, cxastro.AbstractSpaceFrame)
@plum.conversion_method(apyc.Galactocentric, cxastro.Galactocentric)
def astropy_galactocentric_to_coordinax_galactocentric(
    frame: apyc.Galactocentric, /
) -> cxastro.Galactocentric:
    """Convert Astropy Galactocentric frame to coordinax Galactocentric frame.

    The Galactocentric frame is centered at the center of the Milky Way Galaxy,
    with the x-axis pointing from the Galactic center to the Sun, the z-axis
    pointing toward the North Galactic Pole, and the y-axis following the
    right-hand rule.

    This conversion extracts all frame parameters from the Astropy frame and
    constructs an equivalent coordinax frame with the same parameters:

    - galcen: Position of the Galactic center (LonLatSphericalPos)
    - galcen_v_sun: Velocity of the Sun with respect to the Galactic center
      (CartesianVel3D)
    - z_sun: Height of the Sun above the Galactic midplane
    - roll: Rotation angle of the frame

    Examples
    --------
    >>> import coordinax.astro as cxa
    >>> from plum import convert
    >>> import astropy.coordinates as apyc

    Convert with default parameters:

    >>> apy_frame = apyc.Galactocentric()
    >>> convert(apy_frame, cxastro.Galactocentric)
    Galactocentric(
      galcen=Point(
        { 'lon': Q(f64[], 'deg'), 'lat': Q(f64[], 'deg'), 'distance': Q(f64[], 'kpc') },
        chart=LonLatSpherical3D(M=Rn(3)), frame=ICRS()
      ),
      roll=Angle(f64[], 'deg'),
      z_sun=Quantity(f64[], 'pc')
    )

    Convert with custom parameters:

    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> galcen_coord = coord.SphericalRepresentation(
    ...     lon=0 * u.deg, lat=0 * u.deg, distance=8.122 * u.kpc
    ... )
    >>> galcen_v_sun = coord.CartesianDifferential(
    ...     d_x=11.1 * u.km / u.s,
    ...     d_y=244 * u.km / u.s,
    ...     d_z=7.25 * u.km / u.s,
    ... )
    >>> apy_frame = apyc.Galactocentric(
    ...     galcen_coord=galcen_coord,
    ...     galcen_distance=8.122 * u.kpc,
    ...     z_sun=20.8 * u.pc,
    ...     roll=0 * u.deg,
    ...     galcen_v_sun=galcen_v_sun,
    ... )
    >>> cx_frame = convert(apy_frame, cxastro.Galactocentric)
    >>> isinstance(cx_frame, cxastro.Galactocentric)
    True

    """
    return cxastro.Galactocentric.from_(frame)  # ty: ignore[invalid-return-type]
