# mypy: disable-error-code="attr-defined"

"""Interoperability with :mod:`astropy.coordinates`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped
from typing import cast

import astropy.coordinates as apyc
import astropy.units as apyu
from plum import conversion_method, convert

import unxt as u

import coordinax as cx
import coordinax.vecs as cxv

#####################################################################

# =====================================
# Quantity


@conversion_method(cxv.Vector, apyu.Quantity)
def vec_to_q(obj: cxv.Vector, /) -> Shaped[apyu.Quantity, "*batch 3"]:
    """`coordinax.AbstractPos3D` -> `astropy.units.Quantity`.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> from plum import convert
    >>> import astropy.units as apyu

    >>> vec = cxv.CartesianPos3D.from_([1, 2, 3], "km")
    >>> convert(vec, apyu.Quantity)
    <Quantity [1., 2., 3.] km>

    >>> vec = cxv.SphericalPos(r=apyu.Quantity(1, unit="km"),
    ...                       theta=apyu.Quantity(2, unit="deg"),
    ...                       phi=apyu.Quantity(3, unit="deg"))
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.03485167, 0.0018265 , 0.99939084] km>

    >>> vec = cxv.CylindricalPos(rho=apyu.Quantity(1, unit="km"),
    ...                              phi=apyu.Quantity(2, unit="deg"),
    ...                              z=apyu.Quantity(3, unit="m"))
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.99939084, 0.0348995 , 0.003     ] km>

    >>> dif = cxv.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> convert(dif, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s>

    >>> dif2 = cxv.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(dif2, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s2>

    """
    return convert(convert(obj, u.Q), apyu.Quantity)


# =====================================


@conversion_method(cxv.Vector, apyc.CartesianRepresentation)
def convert_vector_to_astropy(obj: cxv.Vector, /) -> apyc.CartesianRepresentation:
    return apyc.CartesianRepresentation(
        x=convert(obj["x"], apyu.Quantity),
        y=convert(obj["y"], apyu.Quantity),
        z=convert(obj["z"], apyu.Quantity),
    )


# =====================================
# CartesianPos3D


@conversion_method(cx.CartesianPos3D, apyc.BaseRepresentation)
@conversion_method(cx.CartesianPos3D, apyc.CartesianRepresentation)
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


@conversion_method(apyc.CartesianRepresentation, cx.CartesianPos3D)
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> cx.CartesianPos3D:
    """`astropy.CartesianRepresentation` -> `coordinax.CartesianPos3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> convert(vec, cx.CartesianPos3D)
    CartesianPos3D(x=Q(1., 'km'), y=Q(2., 'km'), z=Q(3., 'km'))

    """
    return cast("cx.CartesianPos3D", cx.CartesianPos3D.from_(obj))


# =====================================
# CylindricalPos


@conversion_method(cx.vecs.CylindricalPos, apyc.BaseRepresentation)
@conversion_method(cx.vecs.CylindricalPos, apyc.CylindricalRepresentation)
def cyl_to_apycyl(obj: cx.vecs.CylindricalPos, /) -> apyc.CylindricalRepresentation:
    """`coordinax.CylindricalPos` -> `astropy.CylindricalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalPos(rho=u.Q(1, "km"), phi=u.Q(2, "deg"),
    ...                              z=u.Q(3, "m"))
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


@conversion_method(apyc.CylindricalRepresentation, cx.vecs.CylindricalPos)
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> cx.vecs.CylindricalPos:
    """`astropy.CylindricalRepresentation` -> `coordinax.CylindricalPos`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.km, phi=2 * u.deg, z=30 * u.m)
    >>> convert(cyl, cx.vecs.CylindricalPos)
    CylindricalPos(rho=Distance(1., 'km'), phi=Angle(2., 'deg'), z=Q(30., 'm'))

    """
    return cast("cx.vecs.CylindricalPos", cx.vecs.CylindricalPos.from_(obj))


# =====================================
# SphericalPos


@conversion_method(cx.SphericalPos, apyc.BaseRepresentation)
@conversion_method(cx.SphericalPos, apyc.PhysicsSphericalRepresentation)
def sph_to_apysph(obj: cx.SphericalPos, /) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.SphericalPos` -> `astropy.PhysicsSphericalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Q(1, "m"), theta=u.Q(2, "deg"), phi=u.Q(3, "deg"))
    >>> convert(vec, apyc.PhysicsSphericalRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, m)
        (3., 2., 1.)>

    """
    return apyc.PhysicsSphericalRepresentation(
        r=convert(obj.r, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        theta=convert(obj.theta, apyu.Quantity),
    )


@conversion_method(apyc.PhysicsSphericalRepresentation, cx.SphericalPos)
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
    SphericalPos(r=Distance(1., 'km'), theta=Angle(2., 'deg'), phi=Angle(3., 'deg'))

    """
    return cast("cx.SphericalPos", cx.SphericalPos.from_(obj))


# =====================================
# LonLatSphericalPos


@conversion_method(cx.vecs.LonLatSphericalPos, apyc.BaseRepresentation)
@conversion_method(cx.vecs.LonLatSphericalPos, apyc.PhysicsSphericalRepresentation)
def lonlatsph_to_apysph(
    obj: cx.vecs.LonLatSphericalPos, /
) -> apyc.SphericalRepresentation:
    """`coordinax.LonLatSphericalPos` -> `astropy.SphericalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Q(2, "deg"), lat=u.Q(3, "deg"),
    ...                                  distance=u.Q(1, "km"))
    >>> convert(vec, apyc.SphericalRepresentation)
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, km)
        (2., 3., 1.)>

    """
    return apyc.SphericalRepresentation(
        lon=convert(obj.lon, apyu.Quantity),
        lat=convert(obj.lat, apyu.Quantity),
        distance=convert(obj.distance, apyu.Quantity),
    )


@conversion_method(apyc.SphericalRepresentation, cx.vecs.LonLatSphericalPos)
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
      lon=Angle(2., 'deg'), lat=Angle(3., 'deg'), distance=Distance(1., 'km')
    )

    """
    return cast("cx.vecs.LonLatSphericalPos", cx.vecs.LonLatSphericalPos.from_(obj))


# =====================================
# CartesianVel3D


@conversion_method(cx.CartesianVel3D, apyc.BaseDifferential)
@conversion_method(cx.CartesianVel3D, apyc.CartesianDifferential)
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
        d_x=convert(obj.x, apyu.Quantity),
        d_y=convert(obj.y, apyu.Quantity),
        d_z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(apyc.CartesianDifferential, cx.CartesianVel3D)
def apycart3_to_diffcart3(obj: apyc.CartesianDifferential, /) -> cx.CartesianVel3D:
    """`astropy.CartesianDifferential` -> `coordinax.CartesianVel3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> convert(dcart, cx.CartesianVel3D)
    CartesianVel3D(x=Q(1., 'km / s'), y=Q(2., 'km / s'), z=Q(3., 'km / s'))

    """
    return cast("cx.CartesianVel3D", cx.CartesianVel3D.from_(obj))


# =====================================
# CylindricalVel


@conversion_method(cx.vecs.CylindricalVel, apyc.BaseDifferential)
@conversion_method(cx.vecs.CylindricalVel, apyc.CylindricalDifferential)
def diffcyl_to_apycyl(obj: cx.vecs.CylindricalVel, /) -> apyc.CylindricalDifferential:
    """`coordinax.CylindricalVel` -> `astropy.CylindricalDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

    >>> dif = cx.vecs.CylindricalVel(rho=u.Q(1, "km/s"), phi=u.Q(2, "mas/yr"),
    ...                              z=u.Q(3, "km/s"))
    >>> convert(dif, apyc.CylindricalDifferential)
    <CylindricalDifferential (d_rho, d_phi, d_z) in (km / s, mas / yr, km / s)
        (1., 2., 3.)>

    >>> convert(dif, apyc.BaseDifferential)
    <CylindricalDifferential (d_rho, d_phi, d_z) in (km / s, mas / yr, km / s)
        (1., 2., 3.)>

    """
    return apyc.CylindricalDifferential(
        d_rho=convert(obj.rho, apyu.Quantity),
        d_phi=convert(obj.phi, apyu.Quantity),
        d_z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(apyc.CylindricalDifferential, cx.vecs.CylindricalVel)
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
    CylindricalVel(rho=Q(1., 'km / s'), phi=Q(2., 'mas / yr'), z=Q(2., 'km / s'))

    """
    return cast("cx.vecs.CylindricalVel", cx.vecs.CylindricalVel.from_(obj))


# =====================================
# SphericalVel


@conversion_method(cx.SphericalVel, apyc.BaseDifferential)
@conversion_method(cx.SphericalVel, apyc.PhysicsSphericalDifferential)
def diffsph_to_apysph(obj: cx.SphericalVel, /) -> apyc.PhysicsSphericalDifferential:
    """SphericalVel -> `astropy.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.SphericalVel(r=u.Q(1, "km/s"),
    ...                       theta=u.Q(2, "mas/yr"), phi=u.Q(3, "mas/yr"))
    >>> convert(dif, apyc.PhysicsSphericalDifferential)
    <PhysicsSphericalDifferential (d_phi, d_theta, d_r) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    >>> convert(dif, apyc.BaseDifferential)
    <PhysicsSphericalDifferential (d_phi, d_theta, d_r) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    """
    return apyc.PhysicsSphericalDifferential(
        d_r=convert(obj.r, apyu.Quantity),
        d_theta=convert(obj.theta, apyu.Quantity),
        d_phi=convert(obj.phi, apyu.Quantity),
    )


@conversion_method(apyc.PhysicsSphericalDifferential, cx.SphericalVel)
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
    SphericalVel(r=Q(1., 'km / s'), theta=Q(2., 'mas / yr'), phi=Q(3., 'mas / yr'))

    """
    return cast("cx.SphericalVel", cx.SphericalVel.from_(obj))


# =====================================
# LonLatSphericalVel


@conversion_method(cx.vecs.LonLatSphericalVel, apyc.BaseDifferential)
@conversion_method(cx.vecs.LonLatSphericalVel, apyc.SphericalDifferential)
def difflonlatsph_to_apysph(
    obj: cx.vecs.LonLatSphericalVel, /
) -> apyc.SphericalDifferential:
    """LonLatSphericalVel -> `astropy.SphericalVel`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.vecs.LonLatSphericalVel(lat=u.Q(2, "mas/yr"), lon=u.Q(3, "mas/yr"),
    ...                                  distance=u.Q(1, "km/s"))
    >>> convert(dif, apyc.SphericalDifferential)
    <SphericalDifferential (d_lon, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    >>> convert(dif, apyc.BaseDifferential)
    <SphericalDifferential (d_lon, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    """
    return apyc.SphericalDifferential(
        d_distance=convert(obj.distance, apyu.Quantity),
        d_lon=convert(obj.lon, apyu.Quantity),
        d_lat=convert(obj.lat, apyu.Quantity),
    )


@conversion_method(apyc.SphericalDifferential, cx.vecs.LonLatSphericalVel)
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
      lon=Q(3., 'mas / yr'), lat=Q(2., 'mas / yr'), distance=Q(1., 'km / s')
    )

    """
    return cast("cx.vecs.LonLatSphericalVel", cx.vecs.LonLatSphericalVel.from_(obj))


# =====================================
# LonCosLatSphericalVel


@conversion_method(cx.vecs.LonCosLatSphericalVel, apyc.BaseDifferential)
@conversion_method(cx.vecs.LonCosLatSphericalVel, apyc.SphericalCosLatDifferential)
def diffloncoslatsph_to_apysph(
    obj: cx.vecs.LonCosLatSphericalVel, /
) -> apyc.SphericalCosLatDifferential:
    """LonCosLatSphericalVel -> `astropy.SphericalCosLatDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.vecs.LonCosLatSphericalVel(distance=u.Q(1, "km/s"),
    ...                                     lat=u.Q(2, "mas/yr"),
    ...                                     lon_coslat=u.Q(3, "mas/yr"))
    >>> convert(dif, apyc.SphericalCosLatDifferential)
    <SphericalCosLatDifferential (d_lon_coslat, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    >>> convert(dif, apyc.BaseDifferential)
    <SphericalCosLatDifferential (d_lon_coslat, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
        (3., 2., 1.)>

    """  # noqa: E501
    return apyc.SphericalCosLatDifferential(
        d_distance=convert(obj.distance, apyu.Quantity),
        d_lon_coslat=convert(obj.lon_coslat, apyu.Quantity),
        d_lat=convert(obj.lat, apyu.Quantity),
    )


@conversion_method(apyc.SphericalCosLatDifferential, cx.vecs.LonCosLatSphericalVel)
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
      lon_coslat=Q(3., 'mas / yr'),
      lat=Q(2., 'mas / yr'),
      distance=Q(1., 'km / s')
    )

    """
    return cast(
        "cx.vecs.LonCosLatSphericalVel", cx.vecs.LonCosLatSphericalVel.from_(obj)
    )
