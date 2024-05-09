"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


import astropy.coordinates as apyc
import astropy.units as apyu
from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from unxt import Quantity

from .base import Abstract3DVector
from .builtin import (
    Cartesian3DVector,
    CartesianDifferential3D,
    CylindricalDifferential,
    CylindricalVector,
)
from .sphere import (
    LonCosLatSphericalDifferential,
    LonLatSphericalDifferential,
    LonLatSphericalVector,
    SphericalDifferential,
    SphericalVector,
)
from coordinax._utils import dataclass_values, full_shaped

#####################################################################
# Constructors


@Cartesian3DVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[Cartesian3DVector], obj: apyc.BaseRepresentation
) -> Cartesian3DVector:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> cart = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> vec = cx.Cartesian3DVector.constructor(cart)
    >>> vec.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CartesianRepresentation)
    return cls(x=obj.x, y=obj.y, z=obj.z)


@CylindricalVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[CylindricalVector], obj: apyc.BaseRepresentation
) -> CylindricalVector:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg,
    ...                                 z=30 * u.pc)
    >>> vec = cx.CylindricalVector.constructor(cyl)
    >>> vec.rho
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CylindricalRepresentation)
    return cls(rho=obj.rho, phi=obj.phi, z=obj.z)


@SphericalVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[SphericalVector], obj: apyc.BaseRepresentation
) -> SphericalVector:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> vec = cx.SphericalVector.constructor(sph)
    >>> vec.r
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.PhysicsSphericalRepresentation)
    return cls(r=obj.r, theta=obj.theta, phi=obj.phi)


@LonLatSphericalVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[LonLatSphericalVector], obj: apyc.BaseRepresentation
) -> LonLatSphericalVector:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=3 * u.deg, lat=2 * u.deg,
    ...                               distance=1 * u.kpc)
    >>> vec = cx.LonLatSphericalVector.constructor(sph)
    >>> vec.distance
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.SphericalRepresentation)
    return cls(distance=obj.distance, lon=obj.lon, lat=obj.lat)


# -------------------------------------------------------------------


@CartesianDifferential3D.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[CartesianDifferential3D], obj: apyc.CartesianDifferential
) -> CartesianDifferential3D:
    """Construct from a :class:`astropy.coordinates.CartesianDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> dif = cx.CartesianDifferential3D.constructor(dcart)
    >>> dif.d_x
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_x=obj.d_x, d_y=obj.d_y, d_z=obj.d_z)


@CylindricalDifferential.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[CylindricalDifferential], obj: apyc.CylindricalDifferential
) -> CylindricalDifferential:
    """Construct from a :class:`astropy.coordinates.CylindricalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> dif = cx.CylindricalDifferential.constructor(dcyl)
    >>> dif.d_rho
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_rho=obj.d_rho, d_phi=obj.d_phi, d_z=obj.d_z)


@SphericalDifferential.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[SphericalDifferential], obj: apyc.PhysicsSphericalDifferential
) -> SphericalDifferential:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dsph = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                     d_phi=3 * u.mas/u.yr)
    >>> dif = cx.SphericalDifferential.constructor(dsph)
    >>> dif.d_r
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_r=obj.d_r, d_phi=obj.d_phi, d_theta=obj.d_theta)


@LonLatSphericalDifferential.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[LonLatSphericalDifferential], obj: apyc.SphericalDifferential
) -> LonLatSphericalDifferential:
    """Construct from a :class:`astropy.coordinates.SphericalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dsph = SphericalDifferential(d_distance=1 * u.km / u.s,
    ...                              d_lon=2 * u.mas/u.yr,
    ...                              d_lat=3 * u.mas/u.yr)
    >>> dif = cx.LonLatSphericalDifferential.constructor(dsph)
    >>> dif.d_distance
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_distance=obj.d_distance, d_lon=obj.d_lon, d_lat=obj.d_lat)


@LonCosLatSphericalDifferential.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[LonCosLatSphericalDifferential], obj: apyc.SphericalCosLatDifferential
) -> LonCosLatSphericalDifferential:
    """Construct from a :class:`astropy.coordinates.SphericalCosLatDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dsph = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                    d_lon_coslat=2 * u.mas/u.yr,
    ...                                    d_lat=3 * u.mas/u.yr)
    >>> dif = cx.LonCosLatSphericalDifferential.constructor(dsph)
    >>> dif
    LonCosLatSphericalDifferential(
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


@conversion_method(Abstract3DVector, Quantity)  # type: ignore[misc]
def vec_to_q(obj: Abstract3DVector, /) -> Shaped[Quantity["length"], "*batch 3"]:
    """`coordinax.Abstract3DVector` -> `unxt.Quantity`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], unit="kpc"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    >>> vec = cx.SphericalVector(r=Quantity(1, unit="kpc"),
    ...                          theta=Quantity(2, unit="deg"),
    ...                          phi=Quantity(3, unit="deg"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([0.03485167, 0.0018265 , 0.99939084], dtype=float32),
                       unit='kpc')

    >>> vec = cx.CylindricalVector(rho=Quantity(1, unit="kpc"),
    ...                            phi=Quantity(2, unit="deg"),
    ...                            z=Quantity(3, unit="pc"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([0.99939084, 0.0348995 , 0.003     ], dtype=float32),
                       unit='kpc')

    """
    cart = full_shaped(obj.represent_as(Cartesian3DVector))
    return xp.stack(tuple(dataclass_values(cart)), axis=-1)


@conversion_method(CartesianDifferential3D, Quantity)  # type: ignore[misc]
def vec_diff_to_q(
    obj: CartesianDifferential3D, /
) -> Shaped[Quantity["speed"], "*batch 3"]:
    """`coordinax.CartesianDifferential3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.CartesianDifferential3D.constructor(Quantity([1, 2, 3], unit="km/s"))
    >>> convert(dif, Quantity)
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')

    """
    return xp.stack(tuple(dataclass_values(full_shaped(obj))), axis=-1)


#####################################################################
# Astropy


# =====================================
# Cartesian3DVector


@conversion_method(Cartesian3DVector, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(Cartesian3DVector, apyc.CartesianRepresentation)  # type: ignore[misc]
def cart3_to_apycart3(obj: Cartesian3DVector, /) -> apyc.CartesianRepresentation:
    """`coordinax.Cartesian3DVector` -> `astropy.CartesianRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], unit="kpc"))
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


@conversion_method(apyc.CartesianRepresentation, Cartesian3DVector)  # type: ignore[misc]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> Cartesian3DVector:
    """`astropy.CartesianRepresentation` -> `coordinax.Cartesian3DVector`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> convert(vec, cx.Cartesian3DVector)
    Cartesian3DVector(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc")),
      y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc")),
      z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
    )

    """
    return Cartesian3DVector.constructor(obj)


# =====================================
# CylindricalVector


@conversion_method(CylindricalVector, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(CylindricalVector, apyc.CylindricalRepresentation)  # type: ignore[misc]
def cyl_to_apycyl(obj: CylindricalVector, /) -> apyc.CylindricalRepresentation:
    """`coordinax.CylindricalVector` -> `astropy.CylindricalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalVector(rho=Quantity(1, unit="kpc"),
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


@conversion_method(apyc.CylindricalRepresentation, CylindricalVector)  # type: ignore[misc]
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> CylindricalVector:
    """`astropy.CylindricalRepresentation` -> `coordinax.CylindricalVector`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg, z=30 * u.pc)
    >>> convert(cyl, cx.CylindricalVector)
    CylindricalVector(
        rho=Quantity[...](value=f32[], unit=Unit("kpc")),
        phi=Quantity[...](value=f32[], unit=Unit("deg")),
        z=Quantity[...](value=f32[], unit=Unit("pc"))
    )

    """
    return CylindricalVector.constructor(obj)


# =====================================
# SphericalVector


@conversion_method(SphericalVector, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(SphericalVector, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def sph_to_apysph(obj: SphericalVector, /) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.SphericalVector` -> `astropy.PhysicsSphericalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalVector(r=Quantity(1, unit="kpc"),
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


@conversion_method(apyc.PhysicsSphericalRepresentation, SphericalVector)  # type: ignore[misc]
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> SphericalVector:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.SphericalVector`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> convert(sph, cx.SphericalVector)
    SphericalVector(
      r=Distance(value=f32[], unit=Unit("kpc")),
      theta=Quantity[...](value=f32[], unit=Unit("deg")),
      phi=Quantity[...](value=f32[], unit=Unit("deg"))
    )

    """
    return SphericalVector.constructor(obj)


# =====================================
# LonLatSphericalVector


@conversion_method(LonLatSphericalVector, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(LonLatSphericalVector, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def lonlatsph_to_apysph(obj: LonLatSphericalVector, /) -> apyc.SphericalRepresentation:
    """`coordinax.LonLatSphericalVector` -> `astropy.SphericalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalVector(lon=Quantity(2, unit="deg"),
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


@conversion_method(apyc.SphericalRepresentation, LonLatSphericalVector)  # type: ignore[misc]
def apysph_to_lonlatsph(obj: apyc.SphericalRepresentation, /) -> LonLatSphericalVector:
    """`astropy.SphericalRepresentation` -> `coordinax.LonLatSphericalVector`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=2 * u.deg, lat=3 * u.deg,
    ...                               distance=1 * u.kpc)
    >>> convert(sph, cx.LonLatSphericalVector)
    LonLatSphericalVector(
      lon=Quantity[...](value=f32[], unit=Unit("deg")),
      lat=Quantity[...](value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("kpc"))
    )

    """
    return LonLatSphericalVector.constructor(obj)


# =====================================
# CartesianDifferential3D


@conversion_method(CartesianDifferential3D, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(CartesianDifferential3D, apyc.CartesianDifferential)  # type: ignore[misc]
def diffcart3_to_apycart3(
    obj: CartesianDifferential3D, /
) -> apyc.CartesianDifferential:
    """`coordinax.CartesianDifferential3D` -> `astropy.CartesianDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.CartesianDifferential3D.constructor(Quantity([1, 2, 3], unit="km/s"))
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
    apyc.CartesianDifferential, CartesianDifferential3D
)
def apycart3_to_diffcart3(
    obj: apyc.CartesianDifferential, /
) -> CartesianDifferential3D:
    """`astropy.CartesianDifferential` -> `coordinax.CartesianDifferential3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> convert(dcart, cx.CartesianDifferential3D)
    CartesianDifferential3D(
      d_x=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return CartesianDifferential3D.constructor(obj)


# =====================================
# CylindricalDifferential


@conversion_method(CylindricalDifferential, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(CylindricalDifferential, apyc.CylindricalDifferential)  # type: ignore[misc]
def diffcyl_to_apycyl(obj: CylindricalDifferential, /) -> apyc.CylindricalDifferential:
    """`coordinax.CylindricalDifferential` -> `astropy.CylindricalDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

    >>> dif = cx.CylindricalDifferential(d_rho=Quantity(1, unit="km/s"),
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
    apyc.CylindricalDifferential, CylindricalDifferential
)
def apycyl_to_diffcyl(obj: apyc.CylindricalDifferential, /) -> CylindricalDifferential:
    """`astropy.CylindricalDifferential` -> `coordinax.CylindricalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> convert(dcyl, cx.CylindricalDifferential)
    CylindricalDifferential(
      d_rho=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return CylindricalDifferential.constructor(obj)


# =====================================
# SphericalDifferential


@conversion_method(SphericalDifferential, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(SphericalDifferential, apyc.PhysicsSphericalDifferential)  # type: ignore[misc]
def diffsph_to_apysph(
    obj: SphericalDifferential, /
) -> apyc.PhysicsSphericalDifferential:
    """SphericalDifferential -> `astropy.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.SphericalDifferential(d_r=Quantity(1, unit="km/s"),
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
    apyc.PhysicsSphericalDifferential, SphericalDifferential
)
def apysph_to_diffsph(
    obj: apyc.PhysicsSphericalDifferential, /
) -> SphericalDifferential:
    """`astropy.PhysicsSphericalDifferential` -> SphericalDifferential.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dif = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                    d_phi=3 * u.mas/u.yr)
    >>> convert(dif, cx.SphericalDifferential)
    SphericalDifferential(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("mas / yr") )
    )

    """
    return SphericalDifferential.constructor(obj)


# =====================================
# LonLatSphericalDifferential


@conversion_method(LonLatSphericalDifferential, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(LonLatSphericalDifferential, apyc.SphericalDifferential)  # type: ignore[misc]
def difflonlatsph_to_apysph(
    obj: LonLatSphericalDifferential, /
) -> apyc.SphericalDifferential:
    """LonLatSphericalDifferential -> `astropy.SphericalDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.LonLatSphericalDifferential(d_distance=Quantity(1, unit="km/s"),
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
    apyc.SphericalDifferential, LonLatSphericalDifferential
)
def apysph_to_difflonlatsph(
    obj: apyc.SphericalDifferential, /
) -> LonLatSphericalDifferential:
    """`astropy.SphericalDifferential` -> LonLatSphericalDifferential.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dif = SphericalDifferential(d_distance=1 * u.km / u.s, d_lat=2 * u.mas/u.yr,
    ...                             d_lon=3 * u.mas/u.yr)
    >>> convert(dif, cx.LonLatSphericalDifferential)
    LonLatSphericalDifferential(
      d_lon=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return LonLatSphericalDifferential.constructor(obj)


# =====================================
# LonCosLatSphericalDifferential


@conversion_method(LonCosLatSphericalDifferential, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(LonCosLatSphericalDifferential, apyc.SphericalCosLatDifferential)  # type: ignore[misc]
def diffloncoslatsph_to_apysph(
    obj: LonCosLatSphericalDifferential, /
) -> apyc.SphericalCosLatDifferential:
    """LonCosLatSphericalDifferential -> `astropy.SphericalCosLatDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.LonCosLatSphericalDifferential(d_distance=Quantity(1, unit="km/s"),
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
    apyc.SphericalCosLatDifferential, LonCosLatSphericalDifferential
)
def apysph_to_diffloncoslatsph(
    obj: apyc.SphericalCosLatDifferential, /
) -> LonCosLatSphericalDifferential:
    """`astropy.SphericalCosLatDifferential` -> LonCosLatSphericalDifferential.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dif = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                   d_lat=2 * u.mas/u.yr,
    ...                                   d_lon_coslat=3 * u.mas/u.yr)
    >>> convert(dif, cx.LonCosLatSphericalDifferential)
    LonCosLatSphericalDifferential(
      d_lon_coslat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return LonCosLatSphericalDifferential.constructor(obj)
