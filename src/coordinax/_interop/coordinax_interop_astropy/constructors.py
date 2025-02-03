"""Interoperability with :mod:`astropy.coordinates`."""

__all__: list[str] = []


import astropy.coordinates as apyc
import astropy.units as apyu
from plum import convert, dispatch

import unxt as u

import coordinax as cx

# ============================================================================
# From an Astropy object


@dispatch
def vector(obj: apyc.CartesianRepresentation, /) -> cx.CartesianPos3D:
    """Construct from a :class:`astropy.coordinates.CartesianRepresentation`.

    This re-dispatches to :meth:`coordinax.vecs.CartesianPos3D.from_`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> cart = CartesianRepresentation(1, 2, 3, unit="m")
    >>> vec = cx.vector(cart)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1. 2. 3.]>

    """
    return vector(cx.vecs.CartesianPos3D, obj)


@dispatch
def vector(obj: apyc.CylindricalRepresentation, /) -> cx.vecs.CylindricalPos:
    """Construct from a :class:`astropy.coordinates.CylindricalRepresentation`.

    This re-dispatches to :meth:`coordinax.vecs.CylindricalPos.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.km, phi=2 * u.deg,
    ...                                 z=30 * u.m)
    >>> vec = cx.vector(cyl)
    >>> print(vec)
    <CylindricalPos (rho[km], phi[deg], z[m])
        [ 1.  2. 30.]>

    """
    return vector(cx.vecs.CylindricalPos, obj)


@dispatch
def vector(obj: apyc.PhysicsSphericalRepresentation, /) -> cx.SphericalPos:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalRepresentation`.

    This re-dispatches to :meth:`coordinax.vecs.SphericalPos.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.km, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> vec = cx.vector(sph)
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [1. 2. 3.]>

    """
    return vector(cx.SphericalPos, obj)


@dispatch
def vector(obj: apyc.SphericalRepresentation, /) -> cx.vecs.LonLatSphericalPos:
    """Construct from a :class:`astropy.coordinates.SphericalRepresentation`.

    This re-dispatches to :meth:`coordinax.vecs.LonLatSphericalPos.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg,
    ...                               distance=1 * u.km)
    >>> vec = cx.vector(sph)
    >>> print(vec)
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [3. 2. 1.]>

    """
    return vector(cx.vecs.LonLatSphericalPos, obj)


@dispatch
def vector(obj: apyc.UnitSphericalRepresentation) -> cx.vecs.TwoSpherePos:
    """Construct from a :class:`astropy.coordinates.UnitSphericalRepresentation`.

    This re-dispatches to :meth:`coordinax.vecs.TwoSpherePos.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import UnitSphericalRepresentation

    >>> sph = UnitSphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg)
    >>> vec = cx.vector(sph)
    >>> print(vec)
    <TwoSpherePos (theta[deg], phi[deg])
        [2. 3.]>

    """
    return vector(cx.vecs.TwoSpherePos, obj)


@dispatch
def vector(obj: apyc.CartesianDifferential, /) -> cx.CartesianVel3D:
    """Construct from a :class:`astropy.coordinates.CartesianDifferential`.

    This re-dispatches to :meth:`coordinax.vecs.CartesianVel3D.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> dif = cx.vector(dcart)
    >>> print(vec)
    <TwoSpherePos (theta[deg], phi[deg])
        [2. 3.]>

    """
    return vector(cx.CartesianVel3D, obj)


@dispatch
def vector(obj: apyc.CylindricalDifferential, /) -> cx.vecs.CylindricalVel:
    """Construct from a :class:`astropy.coordinates.CylindricalDifferential`.

    This re-dispatches to :meth:`coordinax.vecs.CylindricalVel.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> dif = cx.vector(dcyl)
    >>> print(vec)
    <TwoSpherePos (theta[deg], phi[deg])
        [2. 3.]>

    """
    return vector(cx.vecs.CylindricalVel, obj)


@dispatch
def vector(obj: apyc.PhysicsSphericalDifferential, /) -> cx.SphericalVel:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalDifferential`.

    This re-dispatches to :meth:`coordinax.vecs.SphericalVel.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dsph = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                     d_phi=3 * u.mas/u.yr)
    >>> dif = cx.vector(dsph)
    >>> print(vec)
    <TwoSpherePos (theta[deg], phi[deg])
        [2. 3.]>

    """
    return vector(cx.SphericalVel, obj)


@dispatch
def vector(obj: apyc.SphericalDifferential, /) -> cx.vecs.LonLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalDifferential`.

    This re-dispatches to :meth:`coordinax.vecs.LonLatSphericalVel.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dsph = SphericalDifferential(d_distance=1 * u.km / u.s,
    ...                              d_lon=2 * u.mas/u.yr,
    ...                              d_lat=3 * u.mas/u.yr)
    >>> dif = cx.vector(dsph)
    >>> print(vec)
    <TwoSpherePos (theta[deg], phi[deg])
        [2. 3.]>

    """
    return vector(cx.vecs.LonLatSphericalVel, obj)


@dispatch
def vector(obj: apyc.SphericalCosLatDifferential, /) -> cx.vecs.LonCosLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalCosLatDifferential`.

    This re-dispatches to :meth:`coordinax.vecs.LonCosLatSphericalVel.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dsph = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                    d_lon_coslat=2 * u.mas/u.yr,
    ...                                    d_lat=3 * u.mas/u.yr)
    >>> dif = cx.vector(dsph)
    >>> print(dif)
    <LonCosLatSphericalVel (lon_coslat[mas / yr], lat[mas / yr], distance[km / s])
        [2. 3. 1.]>

    """
    return vector(cx.vecs.LonCosLatSphericalVel, obj)


@dispatch
def vector(obj: apyc.UnitSphericalDifferential) -> cx.vecs.TwoSphereVel:
    """Construct from a :class:`astropy.coordinates.UnitSphericalDifferential`.

    This re-dispatches to :meth:`coordinax.vecs.TwoSphereVel.from_`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import UnitSphericalDifferential

    >>> dsph = UnitSphericalDifferential(d_lon=3 * u.deg/u.s, d_lat=2 * u.deg/u.s)
    >>> vel = cx.vector(dsph)
    >>> print(vel)
    <TwoSphereVel (theta[deg / s], phi[deg / s])
        [2. 3.]>

    """
    return vector(cx.vecs.TwoSphereVel, obj)


# ============================================================================
# From an Astropy coordinate, to a specific vector type


@dispatch
def vector(
    cls: type[cx.CartesianPos3D], obj: apyc.BaseRepresentation, /
) -> cx.CartesianPos3D:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> cart = CartesianRepresentation(1, 2, 3, unit="km")
    >>> vec = cx.CartesianPos3D.from_(cart)
    >>> print(vec)
    <CartesianPos3D (x[km], y[km], z[km])
        [1. 2. 3.]>

    """
    obj = obj.represent_as(apyc.CartesianRepresentation)
    return cls(x=obj.x, y=obj.y, z=obj.z)


@dispatch
def vector(
    cls: type[cx.vecs.CylindricalPos], obj: apyc.BaseRepresentation, /
) -> cx.vecs.CylindricalPos:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.km, phi=2 * u.deg,
    ...                                 z=30 * u.m)
    >>> vec = cx.vecs.CylindricalPos.from_(cyl)
    >>> print(vec)
    <CylindricalPos (rho[km], phi[deg], z[m])
        [ 1.  2. 30.]>

    """
    obj = obj.represent_as(apyc.CylindricalRepresentation)
    return cls(rho=obj.rho, phi=obj.phi, z=obj.z)


@dispatch
def vector(
    cls: type[cx.SphericalPos], obj: apyc.BaseRepresentation, /
) -> cx.SphericalPos:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.km, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> vec = cx.SphericalPos.from_(sph)
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [1. 2. 3.]>

    """
    obj = obj.represent_as(apyc.PhysicsSphericalRepresentation)
    return cls(r=obj.r, theta=obj.theta, phi=obj.phi)


@dispatch
def vector(
    cls: type[cx.vecs.LonLatSphericalPos], obj: apyc.BaseRepresentation, /
) -> cx.vecs.LonLatSphericalPos:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg,
    ...                               distance=1 * u.km)
    >>> vec = cx.vecs.LonLatSphericalPos.from_(sph)
    >>> print(vec)
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [3. 2. 1.]>

    """
    obj = obj.represent_as(apyc.SphericalRepresentation)
    return cls(distance=obj.distance, lon=obj.lon, lat=obj.lat)


@dispatch
def vector(
    cls: type[cx.vecs.TwoSpherePos], obj: apyc.BaseRepresentation, /
) -> cx.vecs.TwoSpherePos:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import UnitSphericalRepresentation

    >>> sph = UnitSphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg)
    >>> vec = cx.vecs.TwoSpherePos.from_(sph)
    >>> print(vec)
    <TwoSpherePos (theta[deg], phi[deg])
        [2. 3.]>

    """
    obj = obj.represent_as(apyc.UnitSphericalRepresentation)
    return cls(phi=obj.lon, theta=obj.lat)


@dispatch
def vector(
    cls: type[cx.CartesianVel3D], obj: apyc.CartesianDifferential, /
) -> cx.CartesianVel3D:
    """Construct from a :class:`astropy.coordinates.CartesianDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> dif = cx.CartesianVel3D.from_(dcart)
    >>> print(vec)
    <TwoSpherePos (theta[deg], phi[deg])
        [2. 3.]>

    """
    return cls(x=obj.d_x, y=obj.d_y, z=obj.d_z)


@dispatch
def vector(
    cls: type[cx.vecs.CylindricalVel], obj: apyc.CylindricalDifferential, /
) -> cx.vecs.CylindricalVel:
    """Construct from a :class:`astropy.coordinates.CylindricalVel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> vec = cx.vecs.CylindricalVel.from_(dcyl)
    >>> print(vec)
    <CylindricalVel (rho[km / s], phi[mas / yr], z[km / s])
        [1. 2. 2.]>

    """
    return cls(rho=obj.d_rho, phi=obj.d_phi, z=obj.d_z)


@dispatch
def vector(
    cls: type[cx.SphericalVel], obj: apyc.PhysicsSphericalDifferential, /
) -> cx.SphericalVel:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dsph = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                     d_phi=3 * u.mas/u.yr)
    >>> vec = cx.SphericalVel.from_(dsph)
    >>> print(vec)
    <SphericalVel (r[km / s], theta[mas / yr], phi[mas / yr])
        [1. 2. 3.]>

    """
    return cls(r=obj.d_r, phi=obj.d_phi, theta=obj.d_theta)


@dispatch
def vector(
    cls: type[cx.vecs.LonLatSphericalVel], obj: apyc.SphericalDifferential, /
) -> cx.vecs.LonLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalVel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dsph = SphericalDifferential(d_distance=1 * u.km / u.s,
    ...                              d_lon=2 * u.mas/u.yr,
    ...                              d_lat=3 * u.mas/u.yr)
    >>> vec = cx.vecs.LonLatSphericalVel.from_(dsph)
    >>> print(vec)
    <LonLatSphericalVel (lon[mas / yr], lat[mas / yr], distance[km / s])
        [2. 3. 1.]>

    """
    return cls(distance=obj.d_distance, lon=obj.d_lon, lat=obj.d_lat)


@dispatch
def vector(
    cls: type[cx.vecs.LonCosLatSphericalVel], obj: apyc.SphericalCosLatDifferential, /
) -> cx.vecs.LonCosLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalCosLatDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dsph = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                    d_lon_coslat=2 * u.mas/u.yr,
    ...                                    d_lat=3 * u.mas/u.yr)
    >>> vec = cx.vecs.LonCosLatSphericalVel.from_(dsph)
    >>> print(vec)
    <LonCosLatSphericalVel (lon_coslat[mas / yr], lat[mas / yr], distance[km / s])
        [2. 3. 1.]>

    """
    return cls(distance=obj.d_distance, lon_coslat=obj.d_lon_coslat, lat=obj.d_lat)


@dispatch
def vector(
    cls: type[cx.vecs.TwoSphereVel], obj: apyc.UnitSphericalDifferential, /
) -> cx.vecs.TwoSphereVel:
    """Construct from a :class:`astropy.coordinates.BaseDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import UnitSphericalDifferential

    >>> sph = UnitSphericalDifferential(d_lon=3 * u.deg/u.s, d_lat=2 * u.deg/u.s)
    >>> vec = cx.vecs.TwoSphereVel.from_(sph)
    >>> print(vec)
    <TwoSphereVel (theta[deg / s], phi[deg / s])
        [2. 3.]>

    """
    return cls(phi=obj.d_lon, theta=obj.d_lat)


#####################################################################


@dispatch
def vector(obj: apyu.Quantity, /) -> cx.vecs.AbstractVector:
    """Construct a vector from an Astropy Quantity.

    The array is expected to have the components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy.units import Quantity
    >>> import coordinax as cx

    >>> vec = cx.vector(Quantity([1, 2, 3], "meter"))
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1. 2. 3.]>

    """
    return vector(convert(obj, u.Quantity))


@dispatch
def vector(
    cls: type[cx.vecs.AbstractVector], obj: apyu.Quantity, /
) -> cx.vecs.AbstractVector:
    """Construct a vector from an Astropy Quantity array.

    The array is expected to have the components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy.units import Quantity
    >>> import coordinax as cx

    >>> xs = Quantity([1, 2, 3], "meter")
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1. 2. 3.]>

    >>> xs = Quantity(jnp.array([[1, 2, 3], [4, 5, 6]]), "meter")
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [[1. 2. 3.]
         [4. 5. 6.]]>

    >>> vec = cx.CartesianVel3D.from_(Quantity([1, 2, 3], "m/s"))
    >>> print(vec)
    <CartesianVel3D (x[m / s], y[m / s], z[m / s])
        [1. 2. 3.]>

    >>> vec = cx.vecs.CartesianAcc3D.from_(Quantity([1, 2, 3], "m/s2"))
    >>> print(vec)
    <CartesianAcc3D (x[m / s2], y[m / s2], z[m / s2])
        [1. 2. 3.]>

    >>> xs = Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
    >>> vec = cx.FourVector.from_(xs)
    >>> print(vec)
    <FourVector (t[m s / km], q=(x[m], y[m], z[m]))
        [0. 1. 2. 3.]>

    >>> xs = Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
    >>> vec = cx.FourVector.from_(xs)
    >>> print(vec)
    <FourVector (t[m s / km], q=(x[m], y[m], z[m]))
        [[0.000e+00 1.000e+00 2.000e+00 3.000e+00]
         [3.336e-05 4.000e+00 5.000e+00 6.000e+00]]>

    """
    return vector(cls, convert(obj, u.Quantity))
