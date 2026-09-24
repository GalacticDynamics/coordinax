"""Interoperability with {mod}`astropy.coordinates`."""

__all__: tuple[str, ...] = ()


from typing import Any, cast

import astropy.coordinates as apyc
import plum

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.vectors as cxv
import coordinaxs.astro as cxastro
from .frames import to_astropy_frame

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


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_unit_spherical_representation(
    cls: type[cxv.Point], obj: apyc.UnitSphericalRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy UnitSphericalRepresentation.

    A direction with no distance, so the point lives on the two-sphere rather
    than in R^3.

    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import UnitSphericalRepresentation
    >>> import astropy.units as apyu

    >>> vec = UnitSphericalRepresentation(lon=90 * apyu.deg, lat=45 * apyu.deg)
    >>> cxv.Point.from_(vec)
    Point(
      {'lon': Q(90., 'deg'), 'lat': Q(45., 'deg')},
      chart=LonLatSphericalTwoSphere(M=Sn(2))
    )

    """
    data = cxc.cdict(obj)
    return cls(data, cxc.lonlat_sph2, frame=cxf.noframe)


@cxv.Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_radial_representation(
    cls: type[cxv.Point], obj: apyc.RadialRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy RadialRepresentation.

    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import RadialRepresentation
    >>> import astropy.units as apyu

    >>> cxv.Point.from_(RadialRepresentation(distance=1 * apyu.kpc))
    Point({'r': Q(1., 'kpc')}, chart=Radial1D(M=Rn(1)))

    """
    data = cxc.cdict(obj)
    return cls(data, cxc.radial1d, frame=cxf.noframe)


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

    A `~coordinax.vectors.Point` is a position, so any velocity the frame
    carries is not part of the result. `~coordinax.vectors.Tangent.from_` takes
    the other half:

    >>> vec = apyc.Galactocentric(
    ...     x=1 * apyu.kpc, y=2 * apyu.kpc, z=3 * apyu.kpc,
    ...     v_x=4 * apyu.km / apyu.s, v_y=5 * apyu.km / apyu.s,
    ...     v_z=6 * apyu.km / apyu.s,
    ... )
    >>> cxv.Tangent.from_(vec)
    Tangent(
      {'x': Q(4., 'km / s'), 'y': Q(5., 'km / s'), 'z': Q(6., 'km / s')},
      chart=Cart3D(M=Rn(3)),
      basis=coord_basis,
      semantic=vel
    )

    """
    if not obj.has_data:
        msg = f"{type(obj).__name__} frame has no data; cannot convert to Point."
        raise ValueError(msg)

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

    # `cast` because ty cannot see through plum's dispatch.
    apy_frame = cast("apyc.BaseCoordinateFrame", to_astropy_frame(obj.frame))
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
# Astropy Data-ful Frames and SkyCoords -> Coordinax Tangent


def _differential(obj: apyc.BaseCoordinateFrame | apyc.SkyCoord, /) -> Any:
    """Pull the velocity differential off an Astropy frame or `SkyCoord`.

    Astropy files a representation's derivatives under a key naming the variable
    they are taken with respect to; ``"s"`` is time. A frame built without
    velocities has no such key.
    """
    differentials = obj.data.differentials
    if "s" not in differentials:
        msg = (
            f"{type(obj).__name__} carries no velocity; "
            "there is nothing to convert to a Tangent."
        )
        raise ValueError(msg)
    return differentials["s"]


@cxv.Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxv.Tangent], obj: apyc.BaseCoordinateFrame, /) -> cxv.Tangent:
    """Construct a Tangent from the velocity of an Astropy frame.

    The position half is `~coordinax.vectors.Point.from_`; this is the other
    half, so a frame carrying both splits into the two coordinax objects.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import coordinax.vectors as cxv

    >>> vec = apyc.Galactocentric(
    ...     x=1 * apyu.kpc, y=2 * apyu.kpc, z=3 * apyu.kpc,
    ...     v_x=4 * apyu.km / apyu.s, v_y=5 * apyu.km / apyu.s,
    ...     v_z=6 * apyu.km / apyu.s,
    ... )
    >>> cxv.Tangent.from_(vec)
    Tangent(
      {'x': Q(4., 'km / s'), 'y': Q(5., 'km / s'), 'z': Q(6., 'km / s')},
      chart=Cart3D(M=Rn(3)),
      basis=coord_basis,
      semantic=vel
    )

    Angular rates come across too, when the frame states them as plain
    ``d_lon`` rather than astropy's ``cos(lat)``-scaled convention:

    >>> vec = apyc.ICRS(
    ...     ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc,
    ...     pm_ra=3 * apyu.mas / apyu.yr, pm_dec=2 * apyu.mas / apyu.yr,
    ...     radial_velocity=10 * apyu.km / apyu.s,
    ...     differential_type=apyc.SphericalDifferential,
    ... )
    >>> cxv.Tangent.from_(vec)
    Tangent(
      {'lon': Q(3., 'mas / yr'), 'lat': Q(2., 'mas / yr'),
       'distance': Q(10., 'km / s')},
      chart=LonLatSpherical3D(M=Rn(3)),
      basis=coord_basis,
      semantic=vel
    )

    A frame without velocities says so rather than handing back a zero:

    >>> try:
    ...     cxv.Tangent.from_(apyc.ICRS(ra=1 * apyu.deg, dec=2 * apyu.deg))
    ... except ValueError as e:
    ...     print(e)
    ICRS carries no velocity; there is nothing to convert to a Tangent.

    """
    del cls
    return plum.convert(_differential(obj), cxv.Tangent)


@cxv.Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_astropy_skycoord_velocity(
    cls: type[cxv.Tangent], obj: apyc.SkyCoord, /
) -> cxv.Tangent:
    """Construct a Tangent from the velocity of an Astropy `SkyCoord`.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import coordinax.vectors as cxv

    A `~astropy.coordinates.SkyCoord` defaults to the ``cos(lat)``-scaled proper
    motion convention, which coordinax has no chart for, so the usual
    astronomical spelling is refused rather than silently reinterpreted:

    >>> sc = apyc.SkyCoord(
    ...     ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc,
    ...     pm_ra_cosdec=3 * apyu.mas / apyu.yr, pm_dec=2 * apyu.mas / apyu.yr,
    ...     radial_velocity=10 * apyu.km / apyu.s,
    ... )
    >>> try:
    ...     cxv.Tangent.from_(sc)
    ... except ValueError as e:
    ...     print(e)
    astropy's SphericalCosLatDifferential is a rate convention ...

    Cartesian velocities carry across unchanged:

    >>> sc = apyc.SkyCoord(
    ...     x=1 * apyu.kpc, y=2 * apyu.kpc, z=3 * apyu.kpc,
    ...     v_x=4 * apyu.km / apyu.s, v_y=5 * apyu.km / apyu.s,
    ...     v_z=6 * apyu.km / apyu.s,
    ...     representation_type="cartesian", differential_type="cartesian",
    ... )
    >>> cxv.Tangent.from_(sc)
    Tangent(
      {'x': Q(4., 'km / s'), 'y': Q(5., 'km / s'), 'z': Q(6., 'km / s')},
      chart=Cart3D(M=Rn(3)),
      basis=coord_basis,
      semantic=vel
    )

    """
    del cls
    return plum.convert(_differential(obj), cxv.Tangent)


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
