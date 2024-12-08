# mypy: disable-error-code="attr-defined"

"""Interoperability with :mod:`astropy.coordinates`."""

__all__: list[str] = []


import astropy.coordinates as apyc
import astropy.units as apyu
from jaxtyping import Shaped
from plum import conversion_method, convert

import unxt as u

import coordinax as cx

#####################################################################

# =====================================
# Quantity


@conversion_method(cx.vecs.AbstractPos3D, apyu.Quantity)  # type: ignore[misc]
def vec_to_q(obj: cx.vecs.AbstractPos3D, /) -> Shaped[apyu.Quantity, "*batch 3"]:
    """`coordinax.AbstractPos3D` -> `astropy.units.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> import astropy.units as apyu

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> convert(vec, apyu.Quantity)
    <Quantity [1., 2., 3.] km>

    >>> vec = cx.SphericalPos(r=apyu.Quantity(1, unit="km"),
    ...                       theta=apyu.Quantity(2, unit="deg"),
    ...                       phi=apyu.Quantity(3, unit="deg"))
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.03485167, 0.0018265 , 0.99939084] km>

    >>> vec = cx.vecs.CylindricalPos(rho=apyu.Quantity(1, unit="km"),
    ...                              phi=apyu.Quantity(2, unit="deg"),
    ...                              z=apyu.Quantity(3, unit="m"))
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.99939084, 0.0348995 , 0.003     ] km>

    """
    return convert(convert(obj, u.Quantity), apyu.Quantity)


@conversion_method(cx.vecs.CartesianAcc3D, apyu.Quantity)  # type: ignore[misc]
@conversion_method(cx.CartesianVel3D, apyu.Quantity)  # type: ignore[misc]
def vec_diff_to_q(
    obj: cx.CartesianVel3D | cx.vecs.CartesianAcc3D, /
) -> Shaped[apyu.Quantity, "*batch 3"]:
    """`coordinax.CartesianVel3D` -> `astropy.units.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from astropy.units import Quantity as AstropyQuantity

    >>> dif = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(dif, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s>

    >>> dif2 = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(dif2, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s2>

    """
    return convert(convert(obj, u.Quantity), apyu.Quantity)


# =====================================
# CartesianPos3D


@conversion_method(cx.CartesianPos3D, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.CartesianPos3D, apyc.CartesianRepresentation)  # type: ignore[misc]
def cart3_to_apycart3(obj: cx.CartesianPos3D, /) -> apyc.CartesianRepresentation:
    """`coordinax.CartesianPos3D` -> `astropy.CartesianRepresentation`.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> convert(vec, apyc.CartesianRepresentation)
    <CartesianRepresentation (x, y, z) in km
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CartesianRepresentation (x, y, z) in km
        (1., 2., 3.)>

    """
    return apyc.CartesianRepresentation(
        x=convert(obj.x, apyu.Quantity),
        y=convert(obj.y, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(apyc.CartesianRepresentation, cx.CartesianPos3D)  # type: ignore[misc]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> cx.CartesianPos3D:
    """`astropy.CartesianRepresentation` -> `coordinax.CartesianPos3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> convert(vec, cx.CartesianPos3D)
    CartesianPos3D(
      x=Quantity[...](value=f32[], unit=Unit("km")),
      y=Quantity[...](value=f32[], unit=Unit("km")),
      z=Quantity[...](value=f32[], unit=Unit("km"))
    )

    """
    return cx.CartesianPos3D.from_(obj)


# =====================================
# CylindricalPos


@conversion_method(cx.vecs.CylindricalPos, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.vecs.CylindricalPos, apyc.CylindricalRepresentation)  # type: ignore[misc]
def cyl_to_apycyl(obj: cx.vecs.CylindricalPos, /) -> apyc.CylindricalRepresentation:
    """`coordinax.CylindricalPos` -> `astropy.CylindricalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalPos(rho=u.Quantity(1, unit="km"),
    ...                              phi=u.Quantity(2, unit="deg"),
    ...                              z=u.Quantity(3, unit="m"))
    >>> convert(vec, apyc.CylindricalRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (km, deg, m)
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (km, deg, m)
        (1., 2., 3.)>

    """
    return apyc.CylindricalRepresentation(
        rho=convert(obj.rho, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(apyc.CylindricalRepresentation, cx.vecs.CylindricalPos)  # type: ignore[misc]
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> cx.vecs.CylindricalPos:
    """`astropy.CylindricalRepresentation` -> `coordinax.CylindricalPos`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.km, phi=2 * u.deg, z=30 * u.m)
    >>> convert(cyl, cx.vecs.CylindricalPos)
    CylindricalPos(
        rho=Quantity[...](value=f32[], unit=Unit("km")),
        phi=Angle(value=f32[], unit=Unit("deg")),
        z=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    return cx.vecs.CylindricalPos.from_(obj)


# =====================================
# SphericalPos


@conversion_method(cx.SphericalPos, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.SphericalPos, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def sph_to_apysph(obj: cx.SphericalPos, /) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.SphericalPos` -> `astropy.PhysicsSphericalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1, unit="m"),
    ...                       theta=u.Quantity(2, unit="deg"),
    ...                       phi=u.Quantity(3, unit="deg"))
    >>> convert(vec, apyc.PhysicsSphericalRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, m)
        (3., 2., 1.)>

    """
    return apyc.PhysicsSphericalRepresentation(
        r=convert(obj.r, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        theta=convert(obj.theta, apyu.Quantity),
    )


@conversion_method(apyc.PhysicsSphericalRepresentation, cx.SphericalPos)  # type: ignore[misc]
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> cx.SphericalPos:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.SphericalPos`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.km, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> convert(sph, cx.SphericalPos)
    SphericalPos(
      r=Distance(value=f32[], unit=Unit("km")),
      theta=Angle(value=f32[], unit=Unit("deg")),
      phi=Angle(value=f32[], unit=Unit("deg"))
    )

    """
    return cx.SphericalPos.from_(obj)


# =====================================
# LonLatSphericalPos


@conversion_method(cx.vecs.LonLatSphericalPos, apyc.BaseRepresentation)  # type: ignore[misc]
@conversion_method(cx.vecs.LonLatSphericalPos, apyc.PhysicsSphericalRepresentation)  # type: ignore[misc]
def lonlatsph_to_apysph(
    obj: cx.vecs.LonLatSphericalPos, /
) -> apyc.SphericalRepresentation:
    """`coordinax.LonLatSphericalPos` -> `astropy.SphericalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Quantity(2, unit="deg"),
    ...                                  lat=u.Quantity(3, unit="deg"),
    ...                                  distance=u.Quantity(1, unit="km"))
    >>> convert(vec, apyc.SphericalRepresentation)
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, km)
        (2., 3., 1.)>

    """
    return apyc.SphericalRepresentation(
        lon=convert(obj.lon, apyu.Quantity),
        lat=convert(obj.lat, apyu.Quantity),
        distance=convert(obj.distance, apyu.Quantity),
    )


@conversion_method(apyc.SphericalRepresentation, cx.vecs.LonLatSphericalPos)  # type: ignore[misc]
def apysph_to_lonlatsph(
    obj: apyc.SphericalRepresentation, /
) -> cx.vecs.LonLatSphericalPos:
    """`astropy.SphericalRepresentation` -> `coordinax.LonLatSphericalPos`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=2 * u.deg, lat=3 * u.deg,
    ...                               distance=1 * u.km)
    >>> convert(sph, cx.vecs.LonLatSphericalPos)
    LonLatSphericalPos(
      lon=Angle(value=f32[], unit=Unit("deg")),
      lat=Angle(value=f32[], unit=Unit("deg")),
      distance=Distance(value=f32[], unit=Unit("km"))
    )

    """
    return cx.vecs.LonLatSphericalPos.from_(obj)


# =====================================
# CartesianVel3D


@conversion_method(cx.CartesianVel3D, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.CartesianVel3D, apyc.CartesianDifferential)  # type: ignore[misc]
def diffcart3_to_apycart3(obj: cx.CartesianVel3D, /) -> apyc.CartesianDifferential:
    """`coordinax.CartesianVel3D` -> `astropy.CartesianDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
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
    apyc.CartesianDifferential, cx.CartesianVel3D
)
def apycart3_to_diffcart3(obj: apyc.CartesianDifferential, /) -> cx.CartesianVel3D:
    """`astropy.CartesianDifferential` -> `coordinax.CartesianVel3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> convert(dcart, cx.CartesianVel3D)
    CartesianVel3D(
      d_x=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return cx.CartesianVel3D.from_(obj)


# =====================================
# CylindricalVel


@conversion_method(cx.vecs.CylindricalVel, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.vecs.CylindricalVel, apyc.CylindricalDifferential)  # type: ignore[misc]
def diffcyl_to_apycyl(obj: cx.vecs.CylindricalVel, /) -> apyc.CylindricalDifferential:
    """`coordinax.CylindricalVel` -> `astropy.CylindricalDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

    >>> dif = cx.vecs.CylindricalVel(d_rho=u.Quantity(1, unit="km/s"),
    ...                              d_phi=u.Quantity(2, unit="mas/yr"),
    ...                              d_z=u.Quantity(3, unit="km/s"))
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
    apyc.CylindricalDifferential, cx.vecs.CylindricalVel
)
def apycyl_to_diffcyl(obj: apyc.CylindricalDifferential, /) -> cx.vecs.CylindricalVel:
    """`astropy.CylindricalVel` -> `coordinax.CylindricalVel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> convert(dcyl, cx.vecs.CylindricalVel)
    CylindricalVel(
      d_rho=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return cx.vecs.CylindricalVel.from_(obj)


# =====================================
# SphericalVel


@conversion_method(cx.SphericalVel, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.SphericalVel, apyc.PhysicsSphericalDifferential)  # type: ignore[misc]
def diffsph_to_apysph(obj: cx.SphericalVel, /) -> apyc.PhysicsSphericalDifferential:
    """SphericalVel -> `astropy.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.SphericalVel(d_r=u.Quantity(1, unit="km/s"),
    ...                       d_theta=u.Quantity(2, unit="mas/yr"),
    ...                       d_phi=u.Quantity(3, unit="mas/yr"))
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
    apyc.PhysicsSphericalDifferential, cx.SphericalVel
)
def apysph_to_diffsph(obj: apyc.PhysicsSphericalDifferential, /) -> cx.SphericalVel:
    """`astropy.PhysicsSphericalDifferential` -> SphericalVel.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dif = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                    d_phi=3 * u.mas/u.yr)
    >>> convert(dif, cx.SphericalVel)
    SphericalVel(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("mas / yr") )
    )

    """
    return cx.SphericalVel.from_(obj)


# =====================================
# LonLatSphericalVel


@conversion_method(cx.vecs.LonLatSphericalVel, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.vecs.LonLatSphericalVel, apyc.SphericalDifferential)  # type: ignore[misc]
def difflonlatsph_to_apysph(
    obj: cx.vecs.LonLatSphericalVel, /
) -> apyc.SphericalDifferential:
    """LonLatSphericalVel -> `astropy.SphericalVel`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.vecs.LonLatSphericalVel(d_distance=u.Quantity(1, unit="km/s"),
    ...                                  d_lat=u.Quantity(2, unit="mas/yr"),
    ...                                  d_lon=u.Quantity(3, unit="mas/yr"))
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
    apyc.SphericalDifferential, cx.vecs.LonLatSphericalVel
)
def apysph_to_difflonlatsph(
    obj: apyc.SphericalDifferential, /
) -> cx.vecs.LonLatSphericalVel:
    """`astropy.coordinates.SphericalDifferential` -> LonLatSphericalVel.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dif = SphericalDifferential(d_distance=1 * u.km / u.s, d_lat=2 * u.mas/u.yr,
    ...                             d_lon=3 * u.mas/u.yr)
    >>> convert(dif, cx.vecs.LonLatSphericalVel)
    LonLatSphericalVel(
      d_lon=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return cx.vecs.LonLatSphericalVel.from_(obj)


# =====================================
# LonCosLatSphericalVel


@conversion_method(cx.vecs.LonCosLatSphericalVel, apyc.BaseDifferential)  # type: ignore[misc]
@conversion_method(cx.vecs.LonCosLatSphericalVel, apyc.SphericalCosLatDifferential)  # type: ignore[misc]
def diffloncoslatsph_to_apysph(
    obj: cx.vecs.LonCosLatSphericalVel, /
) -> apyc.SphericalCosLatDifferential:
    """LonCosLatSphericalVel -> `astropy.SphericalCosLatDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.vecs.LonCosLatSphericalVel(d_distance=u.Quantity(1, unit="km/s"),
    ...                                     d_lat=u.Quantity(2, unit="mas/yr"),
    ...                                     d_lon_coslat=u.Quantity(3, unit="mas/yr"))
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
    apyc.SphericalCosLatDifferential, cx.vecs.LonCosLatSphericalVel
)
def apysph_to_diffloncoslatsph(
    obj: apyc.SphericalCosLatDifferential, /
) -> cx.vecs.LonCosLatSphericalVel:
    """`astropy.SphericalCosLatDifferential` -> LonCosLatSphericalVel.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dif = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                   d_lat=2 * u.mas/u.yr,
    ...                                   d_lon_coslat=3 * u.mas/u.yr)
    >>> convert(dif, cx.vecs.LonCosLatSphericalVel)
    LonCosLatSphericalVel(
      d_lon_coslat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return cx.vecs.LonCosLatSphericalVel.from_(obj)
