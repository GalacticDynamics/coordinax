"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


import astropy.coordinates as apyc
import astropy.units as apyu
from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractPosition3D
from .cartesian import CartesianAcceleration3D, CartesianPosition3D, CartesianVelocity3D
from .cylindrical import CylindricalPosition, CylindricalVelocity
from .sphere import (
    LonCosLatSphericalVelocity,
    LonLatSphericalPosition,
    LonLatSphericalVelocity,
    SphericalPosition,
    SphericalVelocity,
)
from coordinax._utils import dataclass_values, full_shaped

#####################################################################
# Constructors


@CartesianPosition3D.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[CartesianPosition3D], obj: apyc.BaseRepresentation, /
) -> CartesianPosition3D:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> cart = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> vec = cx.CartesianPosition3D.constructor(cart)
    >>> vec.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CartesianRepresentation)
    return cls(x=obj.x, y=obj.y, z=obj.z)


@CylindricalPosition.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[CylindricalPosition], obj: apyc.BaseRepresentation, /
) -> CylindricalPosition:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg,
    ...                                 z=30 * u.pc)
    >>> vec = cx.CylindricalPosition.constructor(cyl)
    >>> vec.rho
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CylindricalRepresentation)
    return cls(rho=obj.rho, phi=obj.phi, z=obj.z)


@SphericalPosition.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[SphericalPosition], obj: apyc.BaseRepresentation, /
) -> SphericalPosition:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> vec = cx.SphericalPosition.constructor(sph)
    >>> vec.r
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.PhysicsSphericalRepresentation)
    return cls(r=obj.r, theta=obj.theta, phi=obj.phi)


@LonLatSphericalPosition.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[LonLatSphericalPosition], obj: apyc.BaseRepresentation, /
) -> LonLatSphericalPosition:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg,
    ...                               distance=1 * u.kpc)
    >>> vec = cx.LonLatSphericalPosition.constructor(sph)
    >>> vec.distance
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.SphericalRepresentation)
    return cls(distance=obj.distance, lon=obj.lon, lat=obj.lat)


# -------------------------------------------------------------------


@CartesianVelocity3D.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[CartesianVelocity3D], obj: apyc.CartesianDifferential, /
) -> CartesianVelocity3D:
    """Construct from a :class:`astropy.coordinates.CartesianDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> dif = cx.CartesianVelocity3D.constructor(dcart)
    >>> dif.d_x
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_x=obj.d_x, d_y=obj.d_y, d_z=obj.d_z)


@CylindricalVelocity.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[CylindricalVelocity], obj: apyc.CylindricalDifferential, /
) -> CylindricalVelocity:
    """Construct from a :class:`astropy.coordinates.CylindricalVelocity`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> dif = cx.CylindricalVelocity.constructor(dcyl)
    >>> dif.d_rho
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_rho=obj.d_rho, d_phi=obj.d_phi, d_z=obj.d_z)


@SphericalVelocity.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[SphericalVelocity], obj: apyc.PhysicsSphericalDifferential, /
) -> SphericalVelocity:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dsph = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                     d_phi=3 * u.mas/u.yr)
    >>> dif = cx.SphericalVelocity.constructor(dsph)
    >>> dif.d_r
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_r=obj.d_r, d_phi=obj.d_phi, d_theta=obj.d_theta)


@LonLatSphericalVelocity.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[LonLatSphericalVelocity], obj: apyc.SphericalDifferential, /
) -> LonLatSphericalVelocity:
    """Construct from a :class:`astropy.coordinates.SphericalVelocity`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dsph = SphericalDifferential(d_distance=1 * u.km / u.s,
    ...                              d_lon=2 * u.mas/u.yr,
    ...                              d_lat=3 * u.mas/u.yr)
    >>> dif = cx.LonLatSphericalVelocity.constructor(dsph)
    >>> dif.d_distance
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_distance=obj.d_distance, d_lon=obj.d_lon, d_lat=obj.d_lat)


@LonCosLatSphericalVelocity.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[LonCosLatSphericalVelocity], obj: apyc.SphericalCosLatDifferential, /
) -> LonCosLatSphericalVelocity:
    """Construct from a :class:`astropy.coordinates.SphericalCosLatDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dsph = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                    d_lon_coslat=2 * u.mas/u.yr,
    ...                                    d_lat=3 * u.mas/u.yr)
    >>> dif = cx.LonCosLatSphericalVelocity.constructor(dsph)
    >>> dif
    LonCosLatSphericalVelocity(
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
# Quantity


@conversion_method(AbstractPosition3D, Quantity)  # type: ignore[misc]
def vec_to_q(obj: AbstractPosition3D, /) -> Shaped[Quantity["length"], "*batch 3"]:
    """`coordinax.AbstractPosition3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], unit="kpc"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    >>> vec = cx.SphericalPosition(r=Quantity(1, unit="kpc"),
    ...                          theta=Quantity(2, unit="deg"),
    ...                          phi=Quantity(3, unit="deg"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([0.03485167, 0.0018265 , 0.99939084], dtype=float32),
                       unit='kpc')

    >>> vec = cx.CylindricalPosition(rho=Quantity(1, unit="kpc"),
    ...                            phi=Quantity(2, unit="deg"),
    ...                            z=Quantity(3, unit="pc"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([0.99939084, 0.0348995 , 0.003     ], dtype=float32),
                       unit='kpc')

    """
    cart = full_shaped(obj.represent_as(CartesianPosition3D))
    return xp.stack(tuple(dataclass_values(cart)), axis=-1)


@conversion_method(CartesianAcceleration3D, Quantity)  # type: ignore[misc]
@conversion_method(CartesianVelocity3D, Quantity)  # type: ignore[misc]
def vec_diff_to_q(obj: CartesianVelocity3D, /) -> Shaped[Quantity["speed"], "*batch 3"]:
    """`coordinax.CartesianVelocity3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.CartesianVelocity3D.constructor(Quantity([1, 2, 3], unit="km/s"))
    >>> convert(dif, Quantity)
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')

    >>> dif2 = cx.CartesianAcceleration3D.constructor(Quantity([1, 2, 3], unit="km/s2"))
    >>> convert(dif2, Quantity)
    Quantity['acceleration'](Array([1., 2., 3.], dtype=float32), unit='km / s2')

    """
    return xp.stack(tuple(dataclass_values(full_shaped(obj))), axis=-1)


#####################################################################
# Astropy


# =====================================
# CartesianPosition3D


@conversion_method(CartesianPosition3D, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(CartesianPosition3D, apyc.CartesianRepresentation)  # type: ignore[misc]
def cart3_to_apycart3(obj: CartesianPosition3D, /) -> apyc.CartesianRepresentation:
    """`coordinax.CartesianPosition3D` -> `astropy.CartesianRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], unit="kpc"))
    >>> convert(vec, apyc.CartesianRepresentation)
    <CartesianRepresentation (x, y, z) in kpc
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CartesianRepresentation (x, y, z) in kpc
        (1., 2., 3.)>

    """
    return apyc.CartesianRepresentation(
        x=convert(obj.x, apyu.Quantity),
        y=convert(obj.y, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(apyc.CartesianRepresentation, CartesianPosition3D)  # type: ignore[misc]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> CartesianPosition3D:
    """`astropy.CartesianRepresentation` -> `coordinax.CartesianPosition3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> convert(vec, cx.CartesianPosition3D)
    CartesianPosition3D(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc")),
      y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc")),
      z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
    )

    """
    return CartesianPosition3D.constructor(obj)


# =====================================
# CylindricalPosition


@conversion_method(CylindricalPosition, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(CylindricalPosition, apyc.CylindricalRepresentation)  # type: ignore[misc]
def cyl_to_apycyl(obj: CylindricalPosition, /) -> apyc.CylindricalRepresentation:
    """`coordinax.CylindricalPosition` -> `astropy.CylindricalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPosition(rho=Quantity(1, unit="kpc"),
    ...                            phi=Quantity(2, unit="deg"),
    ...                            z=Quantity(3, unit="pc"))
    >>> convert(vec, apyc.CylindricalRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (kpc, deg, pc)
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (kpc, deg, pc)
        (1., 2., 3.)>

    """
    return apyc.CylindricalRepresentation(
        rho=convert(obj.rho, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(apyc.CylindricalRepresentation, CylindricalPosition)  # type: ignore[misc]
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> CylindricalPosition:
    """`astropy.CylindricalRepresentation` -> `coordinax.CylindricalPosition`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg, z=30 * u.pc)
    >>> convert(cyl, cx.CylindricalPosition)
    CylindricalPosition(
        rho=Quantity[...](value=f32[], unit=Unit("kpc")),
        phi=Quantity[...](value=f32[], unit=Unit("deg")),
        z=Quantity[...](value=f32[], unit=Unit("pc"))
    )

    """
    return CylindricalPosition.constructor(obj)


# =====================================
# SphericalPosition


@conversion_method(SphericalPosition, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(SphericalPosition, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def sph_to_apysph(obj: SphericalPosition, /) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.SphericalPosition` -> `astropy.PhysicsSphericalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPosition(r=Quantity(1, unit="kpc"),
    ...                          theta=Quantity(2, unit="deg"),
    ...                          phi=Quantity(3, unit="deg"))
    >>> convert(vec, apyc.PhysicsSphericalRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, kpc)
        (3., 2., 1.)>

    """
    return apyc.PhysicsSphericalRepresentation(
        r=convert(obj.r, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        theta=convert(obj.theta, apyu.Quantity),
    )


@conversion_method(apyc.PhysicsSphericalRepresentation, SphericalPosition)  # type: ignore[misc]
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> SphericalPosition:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.SphericalPosition`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> convert(sph, cx.SphericalPosition)
    SphericalPosition(
      r=Distance(value=f32[], unit=Unit("kpc")),
      theta=Quantity[...](value=f32[], unit=Unit("deg")),
      phi=Quantity[...](value=f32[], unit=Unit("deg"))
    )

    """
    return SphericalPosition.constructor(obj)


# =====================================
# LonLatSphericalPosition


@conversion_method(LonLatSphericalPosition, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(LonLatSphericalPosition, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def lonlatsph_to_apysph(
    obj: LonLatSphericalPosition, /
) -> apyc.SphericalRepresentation:
    """`coordinax.LonLatSphericalPosition` -> `astropy.SphericalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPosition(lon=Quantity(2, unit="deg"),
    ...                                lat=Quantity(3, unit="deg"),
    ...                                distance=Quantity(1, unit="kpc"))
    >>> convert(vec, apyc.SphericalRepresentation)
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, kpc)
        (2., 3., 1.)>

    """
    return apyc.SphericalRepresentation(
        lon=convert(obj.lon, apyu.Quantity),
        lat=convert(obj.lat, apyu.Quantity),
        distance=convert(obj.distance, apyu.Quantity),
    )


@conversion_method(apyc.SphericalRepresentation, LonLatSphericalPosition)  # type: ignore[misc]
def apysph_to_lonlatsph(
    obj: apyc.SphericalRepresentation, /
) -> LonLatSphericalPosition:
    """`astropy.SphericalRepresentation` -> `coordinax.LonLatSphericalPosition`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=2 * u.deg, lat=3 * u.deg,
    ...                               distance=1 * u.kpc)
    >>> convert(sph, cx.LonLatSphericalPosition)
    LonLatSphericalPosition(
      lon=Quantity[...](value=f32[], unit=Unit("deg")),
      lat=Quantity[...](value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("kpc"))
    )

    """
    return LonLatSphericalPosition.constructor(obj)


# =====================================
# CartesianVelocity3D


@conversion_method(CartesianVelocity3D, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(CartesianVelocity3D, apyc.CartesianDifferential)  # type: ignore[misc]
def diffcart3_to_apycart3(obj: CartesianVelocity3D, /) -> apyc.CartesianDifferential:
    """`coordinax.CartesianVelocity3D` -> `astropy.CartesianDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.CartesianVelocity3D.constructor(Quantity([1, 2, 3], unit="km/s"))
    >>> convert(dif, apyc.CartesianDifferential)
    <CartesianDifferential (d_x, d_y, d_z) in km / s
        (1., 2., 3.)>

    """
    return apyc.CartesianDifferential(
        d_x=convert(obj.d_x, apyu.Quantity),
        d_y=convert(obj.d_y, apyu.Quantity),
        d_z=convert(obj.d_z, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.CartesianDifferential, CartesianVelocity3D
)
def apycart3_to_diffcart3(obj: apyc.CartesianDifferential, /) -> CartesianVelocity3D:
    """`astropy.CartesianDifferential` -> `coordinax.CartesianVelocity3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> convert(dcart, cx.CartesianVelocity3D)
    CartesianVelocity3D(
      d_x=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return CartesianVelocity3D.constructor(obj)


# =====================================
# CylindricalVelocity


@conversion_method(CylindricalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(CylindricalVelocity, apyc.CylindricalDifferential)  # type: ignore[misc]
def diffcyl_to_apycyl(obj: CylindricalVelocity, /) -> apyc.CylindricalDifferential:
    """`coordinax.CylindricalVelocity` -> `astropy.CylindricalDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

    >>> dif = cx.CylindricalVelocity(d_rho=Quantity(1, unit="km/s"),
    ...                                  d_phi=Quantity(2, unit="mas/yr"),
    ...                                  d_z=Quantity(3, unit="km/s"))
    >>> convert(dif, apyc.CylindricalDifferential)
    <CylindricalDifferential (d_rho, d_phi, d_z) in (km / s, mas / yr, km / s)
        (1., 2., 3.)>

    >>> convert(dif, apyc.BaseDifferential)
    <CylindricalDifferential (d_rho, d_phi, d_z) in (km / s, mas / yr, km / s)
        (1., 2., 3.)>

    """
    return apyc.CylindricalDifferential(
        d_rho=convert(obj.d_rho, apyu.Quantity),
        d_phi=convert(obj.d_phi, apyu.Quantity),
        d_z=convert(obj.d_z, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.CylindricalDifferential, CylindricalVelocity
)
def apycyl_to_diffcyl(obj: apyc.CylindricalDifferential, /) -> CylindricalVelocity:
    """`astropy.CylindricalVelocity` -> `coordinax.CylindricalVelocity`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> convert(dcyl, cx.CylindricalVelocity)
    CylindricalVelocity(
      d_rho=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return CylindricalVelocity.constructor(obj)


# =====================================
# SphericalVelocity


@conversion_method(SphericalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(SphericalVelocity, apyc.PhysicsSphericalDifferential)  # type: ignore[misc]
def diffsph_to_apysph(obj: SphericalVelocity, /) -> apyc.PhysicsSphericalDifferential:
    """SphericalVelocity -> `astropy.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.SphericalVelocity(d_r=Quantity(1, unit="km/s"),
    ...                                d_theta=Quantity(2, unit="mas/yr"),
    ...                                d_phi=Quantity(3, unit="mas/yr"))
    >>> convert(dif, apyc.PhysicsSphericalDifferential)
    <PhysicsSphericalDifferential (d_phi, d_theta, d_r) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    >>> convert(dif, apyc.BaseDifferential)
    <PhysicsSphericalDifferential (d_phi, d_theta, d_r) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    """
    return apyc.PhysicsSphericalDifferential(
        d_r=convert(obj.d_r, apyu.Quantity),
        d_theta=convert(obj.d_theta, apyu.Quantity),
        d_phi=convert(obj.d_phi, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.PhysicsSphericalDifferential, SphericalVelocity
)
def apysph_to_diffsph(obj: apyc.PhysicsSphericalDifferential, /) -> SphericalVelocity:
    """`astropy.PhysicsSphericalDifferential` -> SphericalVelocity.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dif = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                    d_phi=3 * u.mas/u.yr)
    >>> convert(dif, cx.SphericalVelocity)
    SphericalVelocity(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("mas / yr") )
    )

    """
    return SphericalVelocity.constructor(obj)


# =====================================
# LonLatSphericalVelocity


@conversion_method(LonLatSphericalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(LonLatSphericalVelocity, apyc.SphericalDifferential)  # type: ignore[misc]
def difflonlatsph_to_apysph(
    obj: LonLatSphericalVelocity, /
) -> apyc.SphericalDifferential:
    """LonLatSphericalVelocity -> `astropy.SphericalVelocity`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.LonLatSphericalVelocity(d_distance=Quantity(1, unit="km/s"),
    ...                                      d_lat=Quantity(2, unit="mas/yr"),
    ...                                      d_lon=Quantity(3, unit="mas/yr"))
    >>> convert(dif, apyc.SphericalDifferential)
    <SphericalDifferential (d_lon, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    >>> convert(dif, apyc.BaseDifferential)
    <SphericalDifferential (d_lon, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    """
    return apyc.SphericalDifferential(
        d_distance=convert(obj.d_distance, apyu.Quantity),
        d_lon=convert(obj.d_lon, apyu.Quantity),
        d_lat=convert(obj.d_lat, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.SphericalDifferential, LonLatSphericalVelocity
)
def apysph_to_difflonlatsph(
    obj: apyc.SphericalDifferential, /
) -> LonLatSphericalVelocity:
    """`astropy.coordinates.SphericalDifferential` -> LonLatSphericalVelocity.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dif = SphericalDifferential(d_distance=1 * u.km / u.s, d_lat=2 * u.mas/u.yr,
    ...                             d_lon=3 * u.mas/u.yr)
    >>> convert(dif, cx.LonLatSphericalVelocity)
    LonLatSphericalVelocity(
      d_lon=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return LonLatSphericalVelocity.constructor(obj)


# =====================================
# LonCosLatSphericalVelocity


@conversion_method(LonCosLatSphericalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(LonCosLatSphericalVelocity, apyc.SphericalCosLatDifferential)  # type: ignore[misc]
def diffloncoslatsph_to_apysph(
    obj: LonCosLatSphericalVelocity, /
) -> apyc.SphericalCosLatDifferential:
    """LonCosLatSphericalVelocity -> `astropy.SphericalCosLatDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.LonCosLatSphericalVelocity(d_distance=Quantity(1, unit="km/s"),
    ...                                         d_lat=Quantity(2, unit="mas/yr"),
    ...                                         d_lon_coslat=Quantity(3, unit="mas/yr"))
    >>> convert(dif, apyc.SphericalCosLatDifferential)
    <SphericalCosLatDifferential (d_lon_coslat, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    >>> convert(dif, apyc.BaseDifferential)
    <SphericalCosLatDifferential (d_lon_coslat, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    """  # noqa: E501
    return apyc.SphericalCosLatDifferential(
        d_distance=convert(obj.d_distance, apyu.Quantity),
        d_lon_coslat=convert(obj.d_lon_coslat, apyu.Quantity),
        d_lat=convert(obj.d_lat, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.SphericalCosLatDifferential, LonCosLatSphericalVelocity
)
def apysph_to_diffloncoslatsph(
    obj: apyc.SphericalCosLatDifferential, /
) -> LonCosLatSphericalVelocity:
    """`astropy.SphericalCosLatDifferential` -> LonCosLatSphericalVelocity.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dif = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                   d_lat=2 * u.mas/u.yr,
    ...                                   d_lon_coslat=3 * u.mas/u.yr)
    >>> convert(dif, cx.LonCosLatSphericalVelocity)
    LonCosLatSphericalVelocity(
      d_lon_coslat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return LonCosLatSphericalVelocity.constructor(obj)
