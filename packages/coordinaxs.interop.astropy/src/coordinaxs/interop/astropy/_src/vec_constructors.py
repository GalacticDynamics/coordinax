"""Interoperability with {mod}`astropy.coordinates`."""

__all__: tuple[str, ...] = ()


import astropy.coordinates as apyc
import plum

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.vectors as cxv
import coordinaxs.astro as cxastro

##############################################################################
# Representation -> Point


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_cartesian_representation(
    cls: type[cxv.Point], obj: apyc.CartesianRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy CartesianRepresentation.

    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> cxv.Point.from_(vec)
    Point({'x': Q(1., 'km'), 'y': Q(2., 'km'), 'z': Q(3., 'km')}, chart=Cart3D(M=Rn(3)))

    """
    data = cxc.cdict(obj)
    return cls(data, cxc.cart3d, frame=cxf.noframe)


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_cylindrical_representation(
    cls: type[cxv.Point], obj: apyc.CylindricalRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy CylindricalRepresentation.

    >>> import astropy.units as apyu
    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> vec = CylindricalRepresentation(rho=1 * apyu.km, phi=90 * apyu.deg,
    ...                                 z=3 * apyu.km)
    >>> cxv.Point.from_(vec)
    Point(
      {'rho': Q(1., 'km'), 'phi': Q(90., 'deg'), 'z': Q(3., 'km')},
      chart=Cylindrical3D(M=Rn(3))
    )

    """
    data = cxc.cdict(obj)
    return cls(data, cxc.cyl3d, frame=cxf.noframe)


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_physics_spherical_representation(
    cls: type[cxv.Point], obj: apyc.PhysicsSphericalRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy PhysicsSphericalRepresentation.

    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import PhysicsSphericalRepresentation
    >>> import astropy.units as apyu

    >>> vec = PhysicsSphericalRepresentation(
    ...     r=1 * apyu.kpc, theta=45 * apyu.deg, phi=90 * apyu.deg)
    >>> cxv.Point.from_(vec)
    Point(
      {'r': Q(1., 'kpc'), 'theta': Q(45., 'deg'), 'phi': Q(90., 'deg')},
      chart=Spherical3D(M=Rn(3))
    )

    """
    data = cxc.cdict(obj)
    return cls(data, cxc.sph3d, frame=cxf.noframe)


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_spherical_representation(
    cls: type[cxv.Point], obj: apyc.SphericalRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy SphericalRepresentation.

    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import SphericalRepresentation
    >>> import astropy.units as apyu

    >>> vec = SphericalRepresentation(
    ...     lon=90 * apyu.deg, lat=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> cxv.Point.from_(vec)
    Point(
      {'lon': Q(90., 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'kpc')},
      chart=LonLatSpherical3D(M=Rn(3))
    )

    """
    data = cxc.cdict(obj)
    return cls(data, cxc.lonlat_sph3d, frame=cxf.noframe)


##############################################################################
# Astropy Data-ful Frames -> Coordinax Point


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxv.Point], obj: apyc.BaseCoordinateFrame, /) -> cxv.Point:
    """Construct Point from Astropy frame with data.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import coordinax.vectors as cxv

    >>> vec = apyc.ICRS(ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> cxv.Point.from_(vec)
    Point(
      {'lon': Q(90., 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'kpc')},
      chart=LonLatSpherical3D(M=Rn(3)), frame=ICRS()
    )

    >>> vec = apyc.Galactocentric(
    ...     x=1 * apyu.kpc, y=2 * apyu.kpc, z=3 * apyu.kpc
    ... )
    >>> cxv.Point.from_(vec)
    Point(
        {'x': Q(1., 'kpc'), 'y': Q(2., 'kpc'), 'z': Q(3., 'kpc')},
        chart=Cart3D(M=Rn(3)), frame=Galactocentric(...)
    )

    """
    if not obj.has_data:
        raise ValueError("ICRS frame has no data; cannot convert to Point.")

    # Separate the data from the frame
    data = obj.data
    apy_frame = obj.replicate_without_data()

    # Convert the Astropy quantities to coordinax ones.
    data = cxc.cdict(data)
    chart = cxc.guess_chart(data)
    frame = plum.convert(apy_frame, cxastro.AbstractSpaceFrame)

    # Convert the data to a Point
    return cxv.Point(data, chart, frame=frame)


@plum.conversion_method(type_from=apyc.BaseCoordinateFrame, type_to=cxv.Point)
def convert_astropy_frame_with_data_to_cx_point(
    obj: apyc.BaseCoordinateFrame, /
) -> cxv.Point:
    """Convert an Astropy frame with data to a Coordinax Point.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import plum
    >>> import coordinax.vectors as cxv

    >>> vec = apyc.ICRS(ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> plum.convert(vec, cxv.Point)
    Point(
      {'lon': Q(90., 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'kpc')},
      chart=LonLatSpherical3D(M=Rn(3)), frame=ICRS()
    )

    >>> vec = apyc.Galactocentric(
    ...     x=1 * apyu.kpc, y=2 * apyu.kpc, z=3 * apyu.kpc
    ... )
    >>> plum.convert(vec, cxv.Point)
    Point(
        {'x': Q(1., 'kpc'), 'y': Q(2., 'kpc'), 'z': Q(3., 'kpc')},
        chart=Cart3D(M=Rn(3)), frame=Galactocentric(...)
    )

    """
    return cxv.Point.from_(obj)  # ty: ignore[invalid-return-type]


@plum.conversion_method(type_from=cxv.Point, type_to=apyc.BaseCoordinateFrame)
def convert_cx_point_to_astropy_frame_with_data(
    obj: cxv.Point, /
) -> apyc.BaseCoordinateFrame:
    """Convert a Coordinax `Point` (with a frame) to an Astropy frame with data.

    The inverse of :func:`convert_astropy_frame_with_data_to_cx_point`: the
    point's chart data becomes the Astropy representation and the point's frame
    becomes the enclosing Astropy frame. The point must carry a real reference
    frame (not ``noframe``).

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import plum
    >>> import coordinax.vectors as cxv

    >>> vec = apyc.ICRS(ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> point = cxv.Point.from_(vec)
    >>> apy = plum.convert(point, apyc.BaseCoordinateFrame)
    >>> bool(isinstance(apy, apyc.ICRS) and apy.has_data)
    True

    Round-trips back to the original frame with data:

    >>> bool(apy.separation_3d(vec) < 1e-9 * apyu.kpc)
    True

    """
    if obj.frame is cxf.noframe:
        msg = (
            "Point has no reference frame (noframe); cannot convert to an Astropy "
            "coordinate frame. Convert to an Astropy representation instead: "
            "plum.convert(point, astropy.coordinates.BaseRepresentation)."
        )
        raise ValueError(msg)

    apy_frame = plum.convert(obj.frame, apyc.BaseCoordinateFrame)
    representation = plum.convert(obj, apyc.BaseRepresentation)
    return apy_frame.realize_frame(representation)


@plum.conversion_method(type_from=cxv.Point, type_to=apyc.SkyCoord)
def convert_cx_point_to_astropy_skycoord(obj: cxv.Point, /) -> apyc.SkyCoord:
    """Convert a Coordinax `Point` (with a frame) to an Astropy `SkyCoord`.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import plum
    >>> import coordinax.vectors as cxv

    >>> vec = apyc.SkyCoord(ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> point = cxv.Point.from_(vec)
    >>> sc = plum.convert(point, apyc.SkyCoord)
    >>> isinstance(sc, apyc.SkyCoord)
    True
    >>> bool(sc.separation_3d(vec) < 1e-9 * apyu.kpc)
    True

    """
    return apyc.SkyCoord(plum.convert(obj, apyc.BaseCoordinateFrame))


##############################################################################
# Astropy SkyCoord -> Coordinax Point


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_skycoord(cls: type[cxv.Point], obj: apyc.SkyCoord, /) -> cxv.Point:
    """Construct Point from Astropy SkyCoord.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import coordinax.vectors as cxv

    >>> vec = apyc.SkyCoord(ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> cxv.Point.from_(vec)
    Point(
      {'lon': Q(90., 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'kpc')},
      chart=LonLatSpherical3D(M=Rn(3)), frame=ICRS()
    )

    >>> vec = vec.transform_to(apyc.Galactocentric())
    >>> cxv.Point.from_(vec)
    Point(
      {'x': Q(-9.08123957, 'kpc'), 'y': Q(0.21365468, 'kpc'), 'z': Q(0.2056243, 'kpc')},
      chart=Cart3D(M=Rn(3)),
      frame=Galactocentric(
        galcen=Point(
          {
            'lon': Q(266.4051, 'deg'),
            'lat': Q(-28.936175, 'deg'),
            'distance': Q(8.122, 'kpc')
          },
          chart=LonLatSpherical3D(M=Rn(3)),
          frame=ICRS()
        ),
        roll=Angle(0., 'deg'),
        z_sun=Q(20.8, 'pc'),
        galcen_v_sun=Tangent(
          {'x': Q(12.9, 'km / s'), 'y': Q(245.6, 'km / s'), 'z': Q(7.78, 'km / s')},
          chart=Cart3D(M=Rn(3)),
          basis=coord_basis,
          semantic=vel
        )
      )
    )

    """
    # Separate the data from the frame
    data = obj.data
    apy_frame = obj.replicate_without_data()

    # Convert the Astropy quantities to coordinax ones.
    data = cxc.cdict(data)
    chart = cxc.guess_chart(data)
    frame = plum.convert(apy_frame, cxastro.AbstractSpaceFrame)

    # Convert the data to a Point
    return cxv.Point(data, chart, frame)
