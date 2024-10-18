"""Interoperability with :mod:`astropy.coordinates`."""
# mypy: disable-error-code="attr-defined"

__all__: list[str] = []

from collections.abc import Mapping

import astropy.coordinates as apyc
import astropy.units as u
from plum import convert

from unxt import Quantity

import coordinax as cx

#####################################################################


@cx.AbstractVector.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.AbstractVector], obj: Mapping[str, u.Quantity], /
) -> cx.AbstractVector:
    """Construct a vector from a mapping.

    Parameters
    ----------
    cls : type[AbstractVector]
        The vector class.
    obj : Mapping[str, `astropy.units.Quantity`]
        The mapping of components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy.units import Quantity
    >>> import coordinax as cx

    >>> xs = {"x": Quantity(1, "m"), "y": Quantity(2, "m"), "z": Quantity(3, "m")}
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> vec
    CartesianPos3D(
        x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
    )

    >>> xs = {"x": Quantity([1, 2], "m"), "y": Quantity([3, 4], "m"),
    ...       "z": Quantity([5, 6], "m")}
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> vec
    CartesianPos3D(
        x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m"))
    )

    """
    return cls(**obj)


#####################################################################


@cx.AbstractPos3D.from_._f.dispatch(precedence=-1)  # noqa: SLF001
def from_(
    cls: type[cx.AbstractPos3D], obj: apyc.CartesianRepresentation, /
) -> cx.CartesianPos3D:
    """Construct from a :class:`astropy.coordinates.CartesianRepresentation`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> cart = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> vec = cx.AbstractPos3D.from_(cart)
    >>> vec.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    return cx.CartesianPos3D.from_(obj)


@cx.AbstractPos3D.from_._f.dispatch(precedence=-1)  # noqa: SLF001
def from_(
    cls: type[cx.AbstractPos3D], obj: apyc.CylindricalRepresentation, /
) -> cx.CylindricalPos:
    """Construct from a :class:`astropy.coordinates.CylindricalRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg,
    ...                                 z=30 * u.pc)
    >>> vec = cx.AbstractPos3D.from_(cyl)
    >>> vec.rho
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    return cx.CylindricalPos.from_(obj)


@cx.AbstractPos3D.from_._f.dispatch(precedence=-1)  # noqa: SLF001
def from_(
    cls: type[cx.AbstractPos3D], obj: apyc.PhysicsSphericalRepresentation, /
) -> cx.SphericalPos:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> vec = cx.AbstractPos3D.from_(sph)
    >>> vec.r
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    return cx.SphericalPos.from_(obj)


@cx.AbstractPos3D.from_._f.dispatch(precedence=-1)  # noqa: SLF001
def from_(
    cls: type[cx.AbstractPos3D], obj: apyc.SphericalRepresentation, /
) -> cx.LonLatSphericalPos:
    """Construct from a :class:`astropy.coordinates.SphericalRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg,
    ...                               distance=1 * u.kpc)
    >>> vec = cx.AbstractPos3D.from_(sph)
    >>> vec.distance
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    return cx.LonLatSphericalPos.from_(obj)


# -------------------------------------------------------------------


@cx.CartesianPos3D.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.CartesianPos3D], obj: apyc.BaseRepresentation, /
) -> cx.CartesianPos3D:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> cart = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> vec = cx.CartesianPos3D.from_(cart)
    >>> vec.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CartesianRepresentation)
    return cls(x=obj.x, y=obj.y, z=obj.z)


@cx.CylindricalPos.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.CylindricalPos], obj: apyc.BaseRepresentation, /
) -> cx.CylindricalPos:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg,
    ...                                 z=30 * u.pc)
    >>> vec = cx.CylindricalPos.from_(cyl)
    >>> vec.rho
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CylindricalRepresentation)
    return cls(rho=obj.rho, phi=obj.phi, z=obj.z)


@cx.SphericalPos.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.SphericalPos], obj: apyc.BaseRepresentation, /
) -> cx.SphericalPos:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> vec = cx.SphericalPos.from_(sph)
    >>> vec.r
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.PhysicsSphericalRepresentation)
    return cls(r=obj.r, theta=obj.theta, phi=obj.phi)


@cx.LonLatSphericalPos.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.LonLatSphericalPos], obj: apyc.BaseRepresentation, /
) -> cx.LonLatSphericalPos:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg,
    ...                               distance=1 * u.kpc)
    >>> vec = cx.LonLatSphericalPos.from_(sph)
    >>> vec.distance
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.SphericalRepresentation)
    return cls(distance=obj.distance, lon=obj.lon, lat=obj.lat)


#####################################################################


@cx.AbstractVel3D.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.AbstractVel3D], obj: apyc.CartesianDifferential, /
) -> cx.CartesianVel3D:
    """Construct from a :class:`astropy.coordinates.CartesianDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> dif = cx.AbstractVel3D.from_(dcart)
    >>> dif.d_x
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cx.CartesianVel3D.from_(obj)


@cx.AbstractVel3D.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.AbstractVel3D], obj: apyc.CylindricalDifferential, /
) -> cx.CylindricalVel:
    """Construct from a :class:`astropy.coordinates.CylindricalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> dif = cx.AbstractVel3D.from_(dcyl)
    >>> dif.d_rho
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cx.CylindricalVel.from_(obj)


@cx.AbstractVel3D.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.AbstractVel3D], obj: apyc.PhysicsSphericalDifferential, /
) -> cx.SphericalVel:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dsph = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                     d_phi=3 * u.mas/u.yr)
    >>> dif = cx.AbstractVel3D.from_(dsph)
    >>> dif.d_r
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cx.SphericalVel.from_(obj)


@cx.AbstractVel3D.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.AbstractVel3D], obj: apyc.SphericalDifferential, /
) -> cx.LonLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dsph = SphericalDifferential(d_distance=1 * u.km / u.s,
    ...                              d_lon=2 * u.mas/u.yr,
    ...                              d_lat=3 * u.mas/u.yr)
    >>> dif = cx.AbstractVel3D.from_(dsph)
    >>> dif.d_distance
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cx.LonLatSphericalVel.from_(obj)


@cx.AbstractVel3D.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.AbstractVel3D], obj: apyc.SphericalCosLatDifferential, /
) -> cx.LonCosLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalCosLatDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dsph = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                    d_lon_coslat=2 * u.mas/u.yr,
    ...                                    d_lat=3 * u.mas/u.yr)
    >>> dif = cx.AbstractVel3D.from_(dsph)
    >>> dif
    LonCosLatSphericalVel(
      d_lon_coslat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )
    >>> dif.d_distance
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cx.LonCosLatSphericalVel.from_(obj)


# -------------------------------------------------------------------


@cx.CartesianVel3D.from_._f.dispatch  # noqa: SLF001
def from_(
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
    >>> dif.d_x
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_x=obj.d_x, d_y=obj.d_y, d_z=obj.d_z)


@cx.CylindricalVel.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.CylindricalVel], obj: apyc.CylindricalDifferential, /
) -> cx.CylindricalVel:
    """Construct from a :class:`astropy.coordinates.CylindricalVel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> dif = cx.CylindricalVel.from_(dcyl)
    >>> dif.d_rho
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_rho=obj.d_rho, d_phi=obj.d_phi, d_z=obj.d_z)


@cx.SphericalVel.from_._f.dispatch  # noqa: SLF001
def from_(
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
    >>> dif = cx.SphericalVel.from_(dsph)
    >>> dif.d_r
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_r=obj.d_r, d_phi=obj.d_phi, d_theta=obj.d_theta)


@cx.LonLatSphericalVel.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.LonLatSphericalVel], obj: apyc.SphericalDifferential, /
) -> cx.LonLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalVel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dsph = SphericalDifferential(d_distance=1 * u.km / u.s,
    ...                              d_lon=2 * u.mas/u.yr,
    ...                              d_lat=3 * u.mas/u.yr)
    >>> dif = cx.LonLatSphericalVel.from_(dsph)
    >>> dif.d_distance
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_distance=obj.d_distance, d_lon=obj.d_lon, d_lat=obj.d_lat)


@cx.LonCosLatSphericalVel.from_._f.dispatch  # noqa: SLF001
def from_(
    cls: type[cx.LonCosLatSphericalVel], obj: apyc.SphericalCosLatDifferential, /
) -> cx.LonCosLatSphericalVel:
    """Construct from a :class:`astropy.coordinates.SphericalCosLatDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dsph = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                    d_lon_coslat=2 * u.mas/u.yr,
    ...                                    d_lat=3 * u.mas/u.yr)
    >>> dif = cx.LonCosLatSphericalVel.from_(dsph)
    >>> dif
    LonCosLatSphericalVel(
      d_lon_coslat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )
    >>> dif.d_distance
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(
        d_distance=obj.d_distance, d_lon_coslat=obj.d_lon_coslat, d_lat=obj.d_lat
    )


#####################################################################


@cx.AbstractVector.from_._f.dispatch  # noqa: SLF001
def from_(cls: type[cx.AbstractVector], obj: u.Quantity, /) -> cx.AbstractVector:
    """Construct a vector from an Astropy Quantity array.

    The array is expected to have the components as the last dimension.

    Parameters
    ----------
    cls : type[AbstractVector]
        The vector class.
    obj : Quantity[Any, (*#batch, N), "..."]
        The array of components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy.units import Quantity
    >>> import coordinax as cx

    >>> xs = Quantity([1, 2, 3], "meter")
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> vec
    CartesianPos3D(
        x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
    )

    >>> xs = Quantity(jnp.array([[1, 2, 3], [4, 5, 6]]), "meter")
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> vec
    CartesianPos3D(
        x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m"))
    )
    >>> vec.x
    Quantity['length'](Array([1., 4.], dtype=float32), unit='m')

    >>> vec = cx.CartesianVel3D.from_(Quantity([1, 2, 3], "m/s"))
    >>> vec
    CartesianVel3D(
      d_x=Quantity[...]( value=f32[], unit=Unit("m / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("m / s") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("m / s") )
    )

    >>> vec = cx.CartesianAcc3D.from_(Quantity([1, 2, 3], "m/s2"))
    >>> vec
    CartesianAcc3D(
      d2_x=Quantity[...](value=f32[], unit=Unit("m / s2")),
      d2_y=Quantity[...](value=f32[], unit=Unit("m / s2")),
      d2_z=Quantity[...](value=f32[], unit=Unit("m / s2"))
    )

    >>> xs = Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
    >>> vec = cx.FourVector.from_(xs)
    >>> vec
    FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("m s / km")),
        q=CartesianPos3D( ... )
    )

    >>> xs = Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
    >>> vec = cx.FourVector.from_(xs)
    >>> vec
    FourVector(
        t=Quantity[PhysicalType('time')](value=f32[2], unit=Unit("m s / km")),
        q=CartesianPos3D( ... )
    )
    >>> vec.x
    Quantity['length'](Array([1., 4.], dtype=float32), unit='m')

    """
    return cls.from_(convert(obj, Quantity))
