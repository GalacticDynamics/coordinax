"""Interoperability with {mod}`astropy.coordinates` frames.

This module provides bidirectional conversions between coordinaxs.astro reference
frames and Astropy coordinate frames. The conversions preserve all frame
parameters and enable seamless integration between the two libraries.

Supported Frames:

- ICRS: International Celestial Reference System
- Galactic: heliocentric Galactic (l, b) coordinate system
- Galactocentric: Galactic center-based coordinate system

All conversions are implemented using plum's `@plum.conversion_method` decorator,
allowing automatic dispatch when using `plum.convert()`.

Examples
--------
Basic ICRS conversion:

>>> import coordinaxs.astro as cxastro
>>> import astropy.coordinates as apyc
>>> import plum

>>> cx_icrs = cxastro.ICRS()
>>> apy_icrs = plum.convert(cx_icrs, apyc.ICRS)
>>> isinstance(apy_icrs, apyc.ICRS)
True

Galactocentric conversion with custom parameters:

>>> import unxt as u
>>> import coordinax as cx
>>> import coordinax.charts as cxc

>>> galcen = cx.Point.from_(
...     {"lon": u.Q(0, "deg"), "lat": u.Q(0, "deg"),
...      "distance": u.Q(8.122, "kpc")},
...     cxc.lonlat_sph3d,
... )
>>> cx_galcen = cxastro.Galactocentric(
...     galcen=galcen,
...     z_sun=u.Q(20.8, "pc"),
...     roll=u.Q(0, "deg"),
... )
>>> apy_galcen = plum.convert(cx_galcen, apyc.Galactocentric)
>>> isinstance(apy_galcen, apyc.Galactocentric)
True

"""

__all__: tuple[str, ...] = ()


import astropy.coordinates as apyc
import astropy.units as apyu
import plum

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.vectors as cxv
import coordinaxs.astro as cxastro
from .custom_types import CDict


@plum.dispatch
def to_astropy_frame(frame: cxf.AbstractReferenceFrame, /) -> apyc.BaseCoordinateFrame:
    """Return the Astropy frame that corresponds to a coordinax frame.

    `plum.convert` cannot do this job: its ``type[...]`` targets match
    covariantly, so a conversion registered on
    `astropy.coordinates.BaseCoordinateFrame` would claim every Astropy frame
    class and silently answer the wrong one. This has one method per supported
    frame instead, so there is no target class to get wrong.

    This signature is the least specific one, so it is also the fallback: it
    catches every frame the registered pairs miss -- a user-defined
    `AbstractSpaceFrame` subclass, or a frame that is not a space frame at all
    such as `coordinax.frames.alice`. Without it those fail with plum's
    `NotFoundLookupError`, a `LookupError` rather than the `TypeError`
    `plum.convert` raises for an unsupported target, and that leaks through
    ``plum.convert(point, apyc.BaseCoordinateFrame)``. Raising here answers
    with the same exception type and names the offending frame.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.frames as cxf
    >>> import coordinaxs.astro as cxastro
    >>> from coordinaxs.interop.astropy._src.frames import to_astropy_frame

    >>> to_astropy_frame(cxastro.ICRS())
    <ICRS Frame>

    >>> to_astropy_frame(cxastro.Galactic())
    <Galactic Frame>

    A frame with no registered Astropy equivalent:

    >>> try:
    ...     to_astropy_frame(cxf.alice)
    ... except TypeError as e:
    ...     print(e)
    Cannot convert `Alice` to an Astropy frame: no Astropy equivalent is
    registered for this coordinax frame.

    """
    msg = (
        f"Cannot convert `{type(frame).__name__}` to an Astropy frame: no "
        "Astropy equivalent is registered for this coordinax frame."
    )
    raise TypeError(msg)


# =============================================================================
# ICRS


@plum.conversion_method(cxastro.ICRS, apyc.ICRS)
def coordinax_icrs_to_astropy_icrs(frame: cxastro.ICRS, /) -> apyc.ICRS:
    """Convert coordinax ICRS frame to Astropy ICRS frame.

    The ICRS (International Celestial Reference System) frame is a kinematically
    non-rotating coordinate system centered at the solar system barycenter. Both
    coordinax and Astropy implementations have no frame-specific parameters, so
    the conversion is straightforward.

    >>> import coordinaxs.astro as cxastro
    >>> import astropy.coordinates as apyc
    >>> import plum

    >>> cx_frame = cxastro.ICRS()
    >>> plum.convert(cx_frame, apyc.ICRS)
    <ICRS Frame>

    Only the exact Astropy frame class is registered; any other target
    raises rather than silently returning an ICRS frame:

    >>> plum.convert(cx_frame, apyc.FK5)
    Traceback (most recent call last):
        ...
    TypeError: Cannot convert `ICRS()` to `...FK5`.

    """
    return apyc.ICRS()


to_astropy_frame.dispatch(coordinax_icrs_to_astropy_icrs)  # ty: ignore[unresolved-attribute]


@cxf.AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxastro.ICRS], obj: apyc.ICRS, /) -> cxastro.ICRS:
    """Construct from a `astropy.coordinates.ICRS`.

    >>> import coordinaxs.astro as cxastro
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
    >>> import coordinaxs.astro as cxastro
    >>> import plum

    >>> apy_frame = apyc.ICRS()
    >>> plum.convert(apy_frame, cxastro.ICRS)
    ICRS()

    >>> plum.convert(apy_frame, cxastro.AbstractSpaceFrame)
    ICRS()

    """
    return cxastro.ICRS.from_(frame)  # ty: ignore[invalid-return-type]


# =============================================================================
# Galactic


@plum.conversion_method(cxastro.Galactic, apyc.Galactic)
def coordinax_galactic_to_astropy_galactic(frame: cxastro.Galactic, /) -> apyc.Galactic:
    """Convert coordinax Galactic frame to Astropy Galactic frame.

    The (heliocentric) Galactic frame has no frame-specific parameters in
    either library, so the conversion is straightforward.

    >>> import coordinaxs.astro as cxastro
    >>> import astropy.coordinates as apyc
    >>> import plum

    >>> cx_frame = cxastro.Galactic()
    >>> plum.convert(cx_frame, apyc.Galactic)
    <Galactic Frame>

    Only the exact Astropy frame class is registered; any other target
    raises rather than silently returning a Galactic frame:

    >>> plum.convert(cx_frame, apyc.FK5)
    Traceback (most recent call last):
        ...
    TypeError: Cannot convert `Galactic()` to `...FK5`.

    """
    return apyc.Galactic()


to_astropy_frame.dispatch(coordinax_galactic_to_astropy_galactic)  # ty: ignore[unresolved-attribute]


@cxf.AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxastro.Galactic], obj: apyc.Galactic, /) -> cxastro.Galactic:
    """Construct from a `astropy.coordinates.Galactic`.

    >>> import coordinaxs.astro as cxastro
    >>> from plum import convert
    >>> import astropy.coordinates as apyc

    >>> apy_frame = apyc.Galactic()
    >>> cxastro.Galactic.from_(apy_frame)
    Galactic()

    """
    if obj.has_data:
        raise ValueError("Astropy frame must not have data.")
    return cls()


@plum.conversion_method(apyc.Galactic, cxastro.AbstractSpaceFrame)
@plum.conversion_method(apyc.Galactic, cxastro.Galactic)
def astropy_galactic_to_coordinax_galactic(frame: apyc.Galactic, /) -> cxastro.Galactic:
    """Convert Astropy Galactic frame to coordinax Galactic frame.

    >>> import astropy.coordinates as apyc
    >>> import coordinaxs.astro as cxastro
    >>> import plum

    >>> apy_frame = apyc.Galactic()
    >>> plum.convert(apy_frame, cxastro.Galactic)
    Galactic()

    >>> plum.convert(apy_frame, cxastro.AbstractSpaceFrame)
    Galactic()

    """
    return cxastro.Galactic.from_(frame)  # ty: ignore[invalid-return-type]


# =============================================================================
# Galactocentric


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
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinaxs.astro as cxastro
    >>> import plum
    >>> import unxt as u

    Convert with default parameters:

    >>> cx_frame = cxastro.Galactocentric()
    >>> plum.convert(cx_frame, apyc.Galactocentric)
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
        (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

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

    Only the exact Astropy frame class is registered; any other target
    raises rather than silently returning a Galactocentric frame:

    >>> plum.convert(cxastro.Galactocentric(), apyc.FK5)
    Traceback (most recent call last):
        ...
    TypeError: Cannot convert `Galactocentric()` to `...FK5`.

    """  # noqa: E501
    # Convert the galcen position
    # Astropy's own `galcen_coord` default is a `UnitSphericalRepresentation`; the
    # distance goes in `galcen_distance` below, and a frame whose `galcen_coord`
    # carries one cannot be compared or transformed by astropy itself.
    galcen_coord = apyc.ICRS(
        plum.convert(frame.galcen, apyc.SphericalRepresentation).represent_as(
            apyc.UnitSphericalRepresentation
        )
    )

    # Convert the galcen velocity
    galcen_v_sun: apyc.CartesianDifferential = plum.convert(
        frame.galcen_v_sun, apyc.CartesianDifferential
    )

    return apyc.Galactocentric(
        galcen_coord=galcen_coord,
        galcen_distance=plum.convert(frame.galcen["distance"], apyu.Quantity),
        galcen_v_sun=galcen_v_sun,
        z_sun=plum.convert(frame.z_sun, apyu.Quantity),
        roll=plum.convert(frame.roll, apyu.Quantity),
    )


to_astropy_frame.dispatch(coordinax_galactocentric_to_astropy_galactocentric)  # ty: ignore[unresolved-attribute]


@cxf.AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[cxastro.Galactocentric], frame: apyc.Galactocentric, /
) -> cxastro.Galactocentric:
    """Construct from a `astropy.coordinates.Galactocentric`.

    >>> import astropy.coordinates as apyc
    >>> import coordinaxs.astro as cxastro

    >>> apy_gcf = apyc.Galactocentric()
    >>> apy_gcf
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
    (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

    >>> gcf = cxastro.Galactocentric.from_(apy_gcf)
    >>> gcf
    Galactocentric(
      galcen=Point(
        {
          'lon': Q(f64[], 'deg'),
          'lat': Q(f64[], 'deg'),
          'distance': Q(f64[], 'kpc')
        },
        chart=LonLatSpherical3D(M=Rn(3)),
        frame=ICRS()
      ),
      roll=Angle(f64[], 'deg'),
      z_sun=Q(f64[], 'pc'),
      galcen_v_sun=Tangent(
        {'x': Q(f64[], 'km / s'), 'y': Q(f64[], 'km / s'), 'z': Q(f64[], 'km / s')},
        chart=Cart3D(M=Rn(3)),
        basis=coord_basis,
        semantic=vel
      )
    )

    Checking equality

    >>> (gcf.galcen["lon"].ustrip("deg") == apy_gcf.galcen_coord.ra.to_value("deg")
    ...  and gcf.galcen["lat"].ustrip("deg") == apy_gcf.galcen_coord.dec.to_value("deg")
    ...  and gcf.galcen["distance"].ustrip("kpc") == apy_gcf.galcen_distance.to_value("kpc") )
    Array(True, dtype=bool)

    """  # noqa: E501
    if frame.has_data:
        raise ValueError("Astropy frame must not have data.")

    # Convert galcen_coord to Vector with lonlat_sph3d chart and point role
    # galcen_coord is an ICRS coordinate, so access ra/dec from representation
    galcen_data: CDict = {
        "lon": plum.convert(frame.galcen_coord.ra, u.Q),
        "lat": plum.convert(frame.galcen_coord.dec, u.Q),
        "distance": plum.convert(frame.galcen_distance, u.Q),
    }
    galcen = cxv.Point(galcen_data, chart=cxc.lonlat_sph3d, frame=cxastro.icrs)

    # Convert galcen_v_sun to a Cartesian velocity Tangent
    # (astropy stores galcen_v_sun as a CartesianDifferential)
    galcen_v_sun: cxv.Tangent = plum.convert(frame.galcen_v_sun, cxv.Tangent)

    return cxastro.Galactocentric(
        galcen=galcen,
        roll=plum.convert(frame.roll, u.Q),
        z_sun=plum.convert(frame.z_sun, u.Q),
        galcen_v_sun=galcen_v_sun,
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
    >>> import coordinaxs.astro as cxastro
    >>> from plum import convert
    >>> import astropy.coordinates as apyc

    Convert with default parameters:

    >>> apy_frame = apyc.Galactocentric()
    >>> convert(apy_frame, cxastro.Galactocentric)
    Galactocentric(
      galcen=Point(
        {
          'lon': Q(f64[], 'deg'),
          'lat': Q(f64[], 'deg'),
          'distance': Q(f64[], 'kpc')
        },
        chart=LonLatSpherical3D(M=Rn(3)),
        frame=ICRS()
      ),
      roll=Angle(f64[], 'deg'),
      z_sun=Q(f64[], 'pc'),
      galcen_v_sun=Tangent(
        {'x': Q(f64[], 'km / s'), 'y': Q(f64[], 'km / s'), 'z': Q(f64[], 'km / s')},
        chart=Cart3D(M=Rn(3)),
        basis=coord_basis,
        semantic=vel
      )
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
