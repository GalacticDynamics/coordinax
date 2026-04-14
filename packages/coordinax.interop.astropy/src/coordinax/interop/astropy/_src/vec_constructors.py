"""Interoperability with {mod}`astropy.coordinates`."""

__all__: tuple[str, ...] = ()


from plum import convert

import astropy.coordinates as apyc
import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinax.vectors as cxv
from coordinax.internal.custom_types import CDict

##############################################################################
# Representation -> Point


@cxv.Point.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cartesian_representation(
    cls: type[cxv.Point], obj: apyc.CartesianRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy CartesianRepresentation.

    Examples
    --------
    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> cxv.Point.from_(vec)
    Point(
      {'x': Q(1., 'km'), 'y': Q(2., 'km'), 'z': Q(3., 'km')},
      chart=Cart3D(), manifold=EuclideanManifold(ndim=3)
    )

    """
    data: CDict = {k: convert(getattr(obj, k), u.Q) for k in ("x", "y", "z")}
    return cls(data, cxc.cart3d, cxm.euclidean3d)


@cxv.Point.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cylindrical_representation(
    cls: type[cxv.Point], obj: apyc.CylindricalRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy CylindricalRepresentation.

    Examples
    --------
    >>> import astropy.units as apyu
    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> vec = CylindricalRepresentation(rho=1 * apyu.km, phi=90 * apyu.deg,
    ...                                 z=3 * apyu.km)
    >>> cxv.Point.from_(vec)
    Point(
      {'rho': Q(1., 'km'), 'phi': Q(90., 'deg'), 'z': Q(3., 'km')},
      chart=Cylindrical3D(), manifold=EuclideanManifold(ndim=3)
    )

    """
    data: CDict = {k: convert(getattr(obj, k), u.Q) for k in ("rho", "phi", "z")}
    return cls(data, cxc.cyl3d, cxm.euclidean3d)


@cxv.Point.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_physics_spherical_representation(
    cls: type[cxv.Point], obj: apyc.PhysicsSphericalRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy PhysicsSphericalRepresentation.

    Examples
    --------
    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import PhysicsSphericalRepresentation
    >>> import astropy.units as apyu

    >>> vec = PhysicsSphericalRepresentation(
    ...     r=1 * apyu.kpc, theta=45 * apyu.deg, phi=90 * apyu.deg)
    >>> cxv.Point.from_(vec)
    Point(
      {'r': Q(1., 'kpc'), 'theta': Q(45., 'deg'), 'phi': Q(90., 'deg')},
      chart=Spherical3D(), manifold=EuclideanManifold(ndim=3)
    )

    """
    data: CDict = {k: convert(getattr(obj, k), u.Q) for k in ("r", "theta", "phi")}
    return cls(data, cxc.sph3d, cxm.euclidean3d)


@cxv.Point.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_spherical_representation(
    cls: type[cxv.Point], obj: apyc.SphericalRepresentation, /
) -> cxv.Point:
    """Construct Point from Astropy SphericalRepresentation.

    Examples
    --------
    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import SphericalRepresentation
    >>> import astropy.units as apyu

    >>> vec = SphericalRepresentation(
    ...     lon=90 * apyu.deg, lat=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> cxv.Point.from_(vec)
    Point(
      {'lon': Q(90., 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'kpc')},
      chart=LonLatSpherical3D(), manifold=EuclideanManifold(ndim=3)
    )

    """
    data: CDict = {k: convert(getattr(obj, k), u.Q) for k in ("lon", "lat", "distance")}
    return cls(data, cxc.lonlat_sph3d, cxm.euclidean3d)
