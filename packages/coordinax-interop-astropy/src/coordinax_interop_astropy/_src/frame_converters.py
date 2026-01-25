"""Interoperability with :mod:`astropy.coordinates` frames.

This module provides bidirectional conversions between coordinax-astro reference
frames and Astropy coordinate frames. The conversions preserve all frame
parameters and enable seamless integration between the two libraries.

Supported Frames
----------------
- ICRS: International Celestial Reference System
- Galactocentric: Galactic center-based coordinate system

All conversions are implemented using plum's `@conversion_method` decorator,
allowing automatic dispatch when using `plum.convert()`.

Examples
--------
Basic ICRS conversion:

>>> import coordinax_astro as cxa
>>> import astropy.coordinates as apyc
>>> from plum import convert

>>> cx_icrs = cxastro.ICRS()
>>> apy_icrs = convert(cx_icrs, apyc.ICRS)
>>> isinstance(apy_icrs, apyc.ICRS)
True

Galactocentric conversion with custom parameters:

>>> import coordinax as cx
>>> import unxt as u

>>> galcen = cx.Vector.from_(
...     {"lon": u.Q(0, "deg"), "lat": u.Q(0, "deg"),
...      "distance": u.Q(8.122, "kpc")},
...     cx.charts.lonlatsph3d, cx.roles.point
... )
>>> galcen_v_sun = cx.Vector.from_(
...     {"x": u.Q(11.1, "km/s"), "y": u.Q(244, "km/s"),
...      "z": u.Q(7.25, "km/s")},
...     cx.charts.cart3d, cx.roles.phys_vel
... )
>>> cx_galcen = cxastro.Galactocentric(
...     galcen=galcen,
...     z_sun=u.Q(20.8, "pc"),
...     roll=u.Q(0, "deg"),
...     galcen_v_sun=galcen_v_sun,
... )
>>> apy_galcen = convert(cx_galcen, apyc.Galactocentric)
>>> isinstance(apy_galcen, apyc.Galactocentric)
True

Round-trip conversions preserve frame parameters:

>>> cx_result = convert(apy_galcen, cxastro.Galactocentric)
>>> isinstance(cx_result, cxastro.Galactocentric)
True

"""

__all__: tuple[str, ...] = ()


import astropy.coordinates as apyc
import astropy.units as apyu
import equinox as eqx
from plum import conversion_method, convert

import unxt as u

import coordinax as cx
import coordinax.frames as cxf
import coordinax_astro as cxastro

# =============================================================================
# ICRS


@conversion_method(cxastro.ICRS, apyc.ICRS)  # type: ignore[arg-type]
def coordinax_icrs_to_astropy_icrs(frame: cxastro.ICRS, /) -> apyc.ICRS:
    """Convert coordinax ICRS frame to Astropy ICRS frame.

    The ICRS (International Celestial Reference System) frame is a kinematically
    non-rotating coordinate system centered at the solar system barycenter. Both
    coordinax and Astropy implementations have no frame-specific parameters, so
    the conversion is straightforward.

    Parameters
    ----------
    frame : coordinax_astro.ICRS
        The coordinax ICRS frame to convert.

    Returns
    -------
    astropy.coordinates.ICRS
        The equivalent Astropy ICRS frame.

    Examples
    --------
    >>> import coordinax_astro as cxa
    >>> from plum import convert
    >>> import astropy.coordinates as apyc

    >>> cx_frame = cxastro.ICRS()
    >>> apy_frame = convert(cx_frame, apyc.ICRS)
    >>> isinstance(apy_frame, apyc.ICRS)
    True

    """
    return apyc.ICRS()


@cxf.AbstractReferenceFrame.from_.dispatch  # type: ignore[untyped-decorator]
def from_(cls: type[cxastro.ICRS], obj: apyc.ICRS, /) -> cxastro.ICRS:
    """Construct from a `astropy.coordinates.ICRS`.

    Examples
    --------
    >>> import coordinax_astro as cxastro
    >>> from plum import convert
    >>> import astropy.coordinates as apyc

    >>> apy_frame = apyc.ICRS()
    >>> cx_frame = convert(apy_frame, cxastro.ICRS)
    >>> isinstance(cx_frame, cxastro.ICRS)
    True

    >>> cxastro.ICRS.from_(apy_frame)
    ICRS()

    """
    obj = eqx.error_if(obj, obj.has_data, "Astropy frame must not have data.")
    return cls()


@conversion_method(apyc.ICRS, cxastro.ICRS)  # type: ignore[arg-type]
def astropy_icrs_to_coordinax_icrs(frame: apyc.ICRS, /) -> cxastro.ICRS:
    """Convert Astropy ICRS frame to coordinax ICRS frame.

    The ICRS (International Celestial Reference System) frame is a kinematically
    non-rotating coordinate system centered at the solar system barycenter. Both
    coordinax and Astropy implementations have no frame-specific parameters, so
    the conversion is straightforward.

    """
    return cxastro.ICRS.from_(frame)


# =============================================================================
# Galactocentric


@conversion_method(cxastro.Galactocentric, apyc.Galactocentric)  # type: ignore[arg-type]
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

    Parameters
    ----------
    frame : coordinax_astro.Galactocentric
        The coordinax Galactocentric frame to convert.

    Returns
    -------
    astropy.coordinates.Galactocentric
        The equivalent Astropy Galactocentric frame with all parameters preserved.

    Examples
    --------
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx
    >>> import coordinax_astro as cxa
    >>> from plum import convert
    >>> import unxt as u

    Convert with default parameters:

    >>> cx_frame = cxastro.Galactocentric()
    >>> apy_frame = convert(cx_frame, apyc.Galactocentric)
    >>> isinstance(apy_frame, apyc.Galactocentric)
    True

    Convert with custom parameters:

    >>> galcen = cx.Vector.from_(
    ...     {"lon": u.Q(0, "deg"), "lat": u.Q(0, "deg"), "distance": u.Q(8.122, "kpc")},
    ...     cx.charts.lonlatsph3d, cx.roles.point
    ... )
    >>> galcen_v_sun = cx.Vector.from_(
    ...     {"x": u.Q(11.1, "km/s"), "y": u.Q(244, "km/s"), "z": u.Q(7.25, "km/s")},
    ...     cx.charts.cart3d, cx.roles.phys_vel
    ... )
    >>> cx_frame = cxastro.Galactocentric(
    ...     galcen=galcen,
    ...     z_sun=u.Q(20.8, "pc"),
    ...     roll=u.Q(0, "deg"),
    ...     galcen_v_sun=galcen_v_sun,
    ... )
    >>> apy_frame = convert(cx_frame, apyc.Galactocentric)
    >>> isinstance(apy_frame, apyc.Galactocentric)
    True

    """
    # Convert the galcen position
    galcen_coord = apyc.ICRS(convert(frame.galcen, apyc.SphericalRepresentation))

    # Convert the galcen velocity
    galcen_v_sun: apyc.CartesianDifferential = convert(
        frame.galcen_v_sun, apyc.CartesianDifferential
    )

    return apyc.Galactocentric(
        galcen_coord=galcen_coord,
        galcen_distance=convert(frame.galcen["distance"], apyu.Q),
        galcen_v_sun=galcen_v_sun,
        z_sun=convert(frame.z_sun, apyu.Q),
        roll=convert(frame.roll, apyu.Q),
    )


@cxf.AbstractReferenceFrame.from_.dispatch  # type: ignore[untyped-decorator]
def from_(
    cls: type[cxastro.Galactocentric], frame: apyc.Galactocentric, /
) -> cxastro.Galactocentric:
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
        galcen=LonLatSphericalPos(...),
        roll=Quantity(f32[], unit='deg'),
        z_sun=Quantity(f32[], unit='pc'),
        galcen_v_sun=CartesianVel3D(...)
    )

    Checking equality

    >>> (gcf.galcen.lon.ustrip("deg") == apy_gcf.galcen_coord.ra.to_value("deg")
    ...  and gcf.galcen.lat.ustrip("deg") == apy_gcf.galcen_coord.dec.to_value("deg")
    ...  and gcf.galcen.distance.ustrip("kpc") == apy_gcf.galcen_distance.to_value("kpc") )
    Array(True, dtype=bool)


    """  # noqa: E501
    frame = eqx.error_if(frame, frame.has_data, "Astropy frame must not have data.")

    # Convert galcen_coord to Vector with lonlatsph3d chart and point role
    # galcen_coord is an ICRS coordinate, so access ra/dec from representation
    galcen = cx.Vector(
        {
            "lon": convert(frame.galcen_coord.ra, u.Q),
            "lat": convert(frame.galcen_coord.dec, u.Q),
            "distance": convert(frame.galcen_distance, u.Q),
        },
        cx.charts.lonlatsph3d,
        cx.roles.point,
    )

    # Convert galcen_v_sun to CartesianVel3D
    galcen_v_sun = cx.Vector(
        frame.galcen_v_sun, chart=cx.charts.cart3d, role=cx.roles.phys_vel
    )

    return cxastro.Galactocentric(
        galcen=galcen,
        roll=convert(frame.roll, u.Q),
        z_sun=convert(frame.z_sun, u.Q),
        galcen_v_sun=galcen_v_sun,
    )


@conversion_method(apyc.Galactocentric, cxastro.Galactocentric)  # type: ignore[arg-type]
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
    >>> import coordinax_astro as cxa
    >>> from plum import convert
    >>> import astropy.coordinates as apyc

    Convert with default parameters:

    >>> apy_frame = apyc.Galactocentric()
    >>> cx_frame = convert(apy_frame, cxastro.Galactocentric)
    >>> isinstance(cx_frame, cxastro.Galactocentric)
    True

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
    return cxastro.Galactocentric.from_(frame)
