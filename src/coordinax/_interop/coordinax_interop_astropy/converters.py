# mypy: disable-error-code="attr-defined"

"""Interoperability with :mod:`astropy.coordinates`."""

__all__: list[str] = []


import astropy.coordinates as apyc
import astropy.units as u
from jaxtyping import Shaped
from plum import conversion_method, convert

import unxt as ux

import coordinax as cx

#####################################################################

# =====================================
# Quantity


@conversion_method(cx.AbstractPos3D, u.Quantity)  # type: ignore[misc]
def vec_to_q(obj: cx.AbstractPos3D, /) -> Shaped[u.Quantity, "*batch 3"]:
    """`coordinax.AbstractPos3D` -> `astropy.units.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from astropy.units import Quantity

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> convert(vec, Quantity)
    <Quantity [1., 2., 3.] kpc>

    >>> vec = cx.SphericalPos(r=Quantity(1, unit="kpc"),
    ...                            theta=Quantity(2, unit="deg"),
    ...                            phi=Quantity(3, unit="deg"))
    >>> convert(vec, Quantity)
    <Quantity [0.03485167, 0.0018265 , 0.99939084] kpc>

    >>> vec = cx.CylindricalPos(rho=Quantity(1, unit="kpc"),
    ...                              phi=Quantity(2, unit="deg"),
    ...                              z=Quantity(3, unit="pc"))
    >>> convert(vec, Quantity)
    <Quantity [0.99939084, 0.0348995 , 0.003     ] kpc>

    """
    return convert(convert(obj, ux.Quantity), u.Quantity)


@conversion_method(cx.CartesianAcceleration3D, u.Quantity)  # type: ignore[misc]
@conversion_method(cx.CartesianVelocity3D, u.Quantity)  # type: ignore[misc]
def vec_diff_to_q(obj: cx.CartesianVelocity3D, /) -> Shaped[u.Quantity, "*batch 3"]:
    """`coordinax.CartesianVelocity3D` -> `astropy.units.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from astropy.units import Quantity

    >>> dif = cx.CartesianVelocity3D.from_([1, 2, 3], "km/s")
    >>> convert(dif, Quantity)
    <Quantity [1., 2., 3.] km / s>

    >>> dif2 = cx.CartesianAcceleration3D.from_([1, 2, 3], "km/s2")
    >>> convert(dif2, Quantity)
    <Quantity [1., 2., 3.] km / s2>

    """
    return convert(convert(obj, ux.Quantity), u.Quantity)


# =====================================
# CartesianPos3D


@conversion_method(cx.CartesianPos3D, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.CartesianPos3D, apyc.CartesianRepresentation)  # type: ignore[misc]
def cart3_to_apycart3(obj: cx.CartesianPos3D, /) -> apyc.CartesianRepresentation:
    """`coordinax.CartesianPos3D` -> `astropy.CartesianRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> convert(vec, apyc.CartesianRepresentation)
    <CartesianRepresentation (x, y, z) in kpc
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CartesianRepresentation (x, y, z) in kpc
        (1., 2., 3.)>

    """
    return apyc.CartesianRepresentation(
        x=convert(obj.x, u.Quantity),
        y=convert(obj.y, u.Quantity),
        z=convert(obj.z, u.Quantity),
    )


@conversion_method(apyc.CartesianRepresentation, cx.CartesianPos3D)  # type: ignore[misc]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> cx.CartesianPos3D:
    """`astropy.CartesianRepresentation` -> `coordinax.CartesianPos3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> convert(vec, cx.CartesianPos3D)
    CartesianPos3D(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc")),
      y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc")),
      z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
    )

    """
    return cx.CartesianPos3D.from_(obj)


# =====================================
# CylindricalPos


@conversion_method(cx.CylindricalPos, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.CylindricalPos, apyc.CylindricalRepresentation)  # type: ignore[misc]
def cyl_to_apycyl(obj: cx.CylindricalPos, /) -> apyc.CylindricalRepresentation:
    """`coordinax.CylindricalPos` -> `astropy.CylindricalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(rho=Quantity(1, unit="kpc"),
    ...                         phi=Quantity(2, unit="deg"),
    ...                         z=Quantity(3, unit="pc"))
    >>> convert(vec, apyc.CylindricalRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (kpc, deg, pc)
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (kpc, deg, pc)
        (1., 2., 3.)>

    """
    return apyc.CylindricalRepresentation(
        rho=convert(obj.rho, u.Quantity),
        phi=convert(obj.phi, u.Quantity),
        z=convert(obj.z, u.Quantity),
    )


@conversion_method(apyc.CylindricalRepresentation, cx.CylindricalPos)  # type: ignore[misc]
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> cx.CylindricalPos:
    """`astropy.CylindricalRepresentation` -> `coordinax.CylindricalPos`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg, z=30 * u.pc)
    >>> convert(cyl, cx.CylindricalPos)
    CylindricalPos(
        rho=Quantity[...](value=f32[], unit=Unit("kpc")),
        phi=Quantity[...](value=f32[], unit=Unit("deg")),
        z=Quantity[...](value=f32[], unit=Unit("pc"))
    )

    """
    return cx.CylindricalPos.from_(obj)


# =====================================
# SphericalPos


@conversion_method(cx.SphericalPos, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.SphericalPos, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def sph_to_apysph(obj: cx.SphericalPos, /) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.SphericalPos` -> `astropy.PhysicsSphericalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=Quantity(1, unit="kpc"),
    ...                       theta=Quantity(2, unit="deg"),
    ...                       phi=Quantity(3, unit="deg"))
    >>> convert(vec, apyc.PhysicsSphericalRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, kpc)
        (3., 2., 1.)>

    """
    return apyc.PhysicsSphericalRepresentation(
        r=convert(obj.r, u.Quantity),
        phi=convert(obj.phi, u.Quantity),
        theta=convert(obj.theta, u.Quantity),
    )


@conversion_method(apyc.PhysicsSphericalRepresentation, cx.SphericalPos)  # type: ignore[misc]
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> cx.SphericalPos:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.SphericalPos`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> convert(sph, cx.SphericalPos)
    SphericalPos(
      r=Distance(value=f32[], unit=Unit("kpc")),
      theta=Quantity[...](value=f32[], unit=Unit("deg")),
      phi=Quantity[...](value=f32[], unit=Unit("deg"))
    )

    """
    return cx.SphericalPos.from_(obj)


# =====================================
# LonLatSphericalPos


@conversion_method(cx.LonLatSphericalPos, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.LonLatSphericalPos, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def lonlatsph_to_apysph(obj: cx.LonLatSphericalPos, /) -> apyc.SphericalRepresentation:
    """`coordinax.LonLatSphericalPos` -> `astropy.SphericalRepresentation`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=Quantity(2, unit="deg"),
    ...                             lat=Quantity(3, unit="deg"),
    ...                             distance=Quantity(1, unit="kpc"))
    >>> convert(vec, apyc.SphericalRepresentation)
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, kpc)
        (2., 3., 1.)>

    """
    return apyc.SphericalRepresentation(
        lon=convert(obj.lon, u.Quantity),
        lat=convert(obj.lat, u.Quantity),
        distance=convert(obj.distance, u.Quantity),
    )


@conversion_method(apyc.SphericalRepresentation, cx.LonLatSphericalPos)  # type: ignore[misc]
def apysph_to_lonlatsph(obj: apyc.SphericalRepresentation, /) -> cx.LonLatSphericalPos:
    """`astropy.SphericalRepresentation` -> `coordinax.LonLatSphericalPos`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=2 * u.deg, lat=3 * u.deg,
    ...                               distance=1 * u.kpc)
    >>> convert(sph, cx.LonLatSphericalPos)
    LonLatSphericalPos(
      lon=Quantity[...](value=f32[], unit=Unit("deg")),
      lat=Quantity[...](value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("kpc"))
    )

    """
    return cx.LonLatSphericalPos.from_(obj)


# =====================================
# CartesianVelocity3D


@conversion_method(cx.CartesianVelocity3D, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.CartesianVelocity3D, apyc.CartesianDifferential)  # type: ignore[misc]
def diffcart3_to_apycart3(obj: cx.CartesianVelocity3D, /) -> apyc.CartesianDifferential:
    """`coordinax.CartesianVelocity3D` -> `astropy.CartesianDifferential`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dif = cx.CartesianVelocity3D.from_([1, 2, 3], "km/s")
    >>> convert(dif, apyc.CartesianDifferential)
    <CartesianDifferential (d_x, d_y, d_z) in km / s
        (1., 2., 3.)>

    """
    return apyc.CartesianDifferential(
        d_x=convert(obj.d_x, u.Quantity),
        d_y=convert(obj.d_y, u.Quantity),
        d_z=convert(obj.d_z, u.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.CartesianDifferential, cx.CartesianVelocity3D
)
def apycart3_to_diffcart3(obj: apyc.CartesianDifferential, /) -> cx.CartesianVelocity3D:
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
    return cx.CartesianVelocity3D.from_(obj)


# =====================================
# CylindricalVelocity


@conversion_method(cx.CylindricalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.CylindricalVelocity, apyc.CylindricalDifferential)  # type: ignore[misc]
def diffcyl_to_apycyl(obj: cx.CylindricalVelocity, /) -> apyc.CylindricalDifferential:
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
        d_rho=convert(obj.d_rho, u.Quantity),
        d_phi=convert(obj.d_phi, u.Quantity),
        d_z=convert(obj.d_z, u.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.CylindricalDifferential, cx.CylindricalVelocity
)
def apycyl_to_diffcyl(obj: apyc.CylindricalDifferential, /) -> cx.CylindricalVelocity:
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
    return cx.CylindricalVelocity.from_(obj)


# =====================================
# SphericalVelocity


@conversion_method(cx.SphericalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.SphericalVelocity, apyc.PhysicsSphericalDifferential)  # type: ignore[misc]
def diffsph_to_apysph(
    obj: cx.SphericalVelocity, /
) -> apyc.PhysicsSphericalDifferential:
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
        d_r=convert(obj.d_r, u.Quantity),
        d_theta=convert(obj.d_theta, u.Quantity),
        d_phi=convert(obj.d_phi, u.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.PhysicsSphericalDifferential, cx.SphericalVelocity
)
def apysph_to_diffsph(
    obj: apyc.PhysicsSphericalDifferential, /
) -> cx.SphericalVelocity:
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
    return cx.SphericalVelocity.from_(obj)


# =====================================
# LonLatSphericalVelocity


@conversion_method(cx.LonLatSphericalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.LonLatSphericalVelocity, apyc.SphericalDifferential)  # type: ignore[misc]
def difflonlatsph_to_apysph(
    obj: cx.LonLatSphericalVelocity, /
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
        d_distance=convert(obj.d_distance, u.Quantity),
        d_lon=convert(obj.d_lon, u.Quantity),
        d_lat=convert(obj.d_lat, u.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.SphericalDifferential, cx.LonLatSphericalVelocity
)
def apysph_to_difflonlatsph(
    obj: apyc.SphericalDifferential, /
) -> cx.LonLatSphericalVelocity:
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
    return cx.LonLatSphericalVelocity.from_(obj)


# =====================================
# LonCosLatSphericalVelocity


@conversion_method(cx.LonCosLatSphericalVelocity, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.LonCosLatSphericalVelocity, apyc.SphericalCosLatDifferential)  # type: ignore[misc]
def diffloncoslatsph_to_apysph(
    obj: cx.LonCosLatSphericalVelocity, /
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
        d_distance=convert(obj.d_distance, u.Quantity),
        d_lon_coslat=convert(obj.d_lon_coslat, u.Quantity),
        d_lat=convert(obj.d_lat, u.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    apyc.SphericalCosLatDifferential, cx.LonCosLatSphericalVelocity
)
def apysph_to_diffloncoslatsph(
    obj: apyc.SphericalCosLatDifferential, /
) -> cx.LonCosLatSphericalVelocity:
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
    return cx.LonCosLatSphericalVelocity.from_(obj)
