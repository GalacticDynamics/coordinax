# mypy: disable-error-code="attr-defined"

"""Interoperability with :mod:`astropy.coordinates`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped

import astropy.coordinates as apyc
import astropy.units as apyu
from plum import conversion_method, convert

import unxt as u

import coordinax as cx
import coordinax.vecs as cxv
from coordinax import r

#####################################################################

# =====================================
# Quantity


@conversion_method(cxv.Vector, apyu.Quantity)  # type: ignore[arg-type]
def vec_to_q(obj: cxv.Vector, /) -> Shaped[apyu.Quantity, "*batch 3"]:
    """`coordinax.AbstractPos3D` -> `astropy.units.Quantity`.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> from plum import convert
    >>> import astropy.units as apyu

    >>> vec = cxv.Cart3D.from_([1, 2, 3], "km")
    >>> convert(vec, apyu.Quantity)
    <Quantity [1., 2., 3.] km>

    >>> vec = cxv.Spherical3D(r=apyu.Quantity(1, unit="km"),
    ...                       theta=apyu.Quantity(2, unit="deg"),
    ...                       phi=apyu.Quantity(3, unit="deg"))
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.03485167, 0.0018265 , 0.99939084] km>

    >>> vec = cxv.Cylindrical3D(rho=apyu.Quantity(1, unit="km"),
    ...                              phi=apyu.Quantity(2, unit="deg"),
    ...                              z=apyu.Quantity(3, unit="m"))
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.99939084, 0.0348995 , 0.003     ] km>

    >>> dif = cxv.CartVel3D.from_([1, 2, 3], "km/s")
    >>> convert(dif, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s>

    >>> dif2 = cxv.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> convert(dif2, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s2>

    """
    return convert(convert(obj, u.Q), apyu.Quantity)


# =====================================
# Cart3D


@conversion_method(cxv.Vector, apyc.CartesianRepresentation)  # type: ignore[arg-type]
def convert_vector_to_astropy(obj: cxv.Vector, /) -> apyc.CartesianRepresentation:
    """`coordinax.Vector` -> `astropy.CartesianRepresentation`.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "km")
    >>> convert(vec, apyc.CartesianRepresentation)
    <CartesianRepresentation (x, y, z) in km
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CartesianRepresentation (x, y, z) in km
        (1., 2., 3.)>

    """
    # Convert to Cart3D first.  # TODO: more generic
    obj = vconvert(r.cart3d, obj)
    # Now convert to Astropy.
    return apyc.CartesianRepresentation(
        x=convert(obj["x"], apyu.Quantity),
        y=convert(obj["y"], apyu.Quantity),
        z=convert(obj["z"], apyu.Quantity),
    )


@conversion_method(apyc.CartesianRepresentation, cx.Vector)  # type: ignore[arg-type]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> cx.Vector[r.Cart3D]:
    """`astropy.CartesianRepresentation` -> `coordinax.Cart3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> convert(vec, cx.Vector)
    Cart3D(x=Q(1., 'km'), y=Q(2., 'km'), z=Q(3., 'km'))

    """
    return cx.Vector[r.Cart3D].from_(obj)


# =====================================
# Cylindrical3D


@conversion_method(cxv.Vector, apyc.BaseRepresentation)
@conversion_method(cxv.Vector, apyc.CylindricalRepresentation)  # type: ignore[arg-type]
def cyl_to_apycyl(
    obj: cxv.Vector[r.Cylindrical3D], /
) -> apyc.CylindricalRepresentation:
    """`coordinax.Cylindrical3D` -> `astropy.CylindricalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.Cylindrical3D(rho=u.Q(1, "km"), phi=u.Q(2, "deg"),
    ...                              z=u.Q(3, "m"))
    >>> convert(vec, apyc.CylindricalRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (km, deg, m)
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (km, deg, m)
        (1., 2., 3.)>

    """
    return apyc.CylindricalRepresentation(
        rho=convert(obj["rho"], apyu.Quantity),
        phi=convert(obj["phi"], apyu.Quantity),
        z=convert(obj["z"], apyu.Quantity),
    )


@conversion_method(apyc.CylindricalRepresentation, cxv.Vector)  # type: ignore[arg-type]
def apycyl_to_cyl(
    obj: apyc.CylindricalRepresentation, /
) -> cxv.Vector[r.Cylindrical3D]:
    """`astropy.CylindricalRepresentation` -> `coordinax.Cylindrical3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.km, phi=2 * u.deg, z=30 * u.m)
    >>> convert(cyl, cx.vecs.Cylindrical3D)
    Cylindrical3D(rho=Distance(1., 'km'), phi=Angle(2., 'deg'), z=Q(30., 'm'))

    """
    return cxv.Vector[r.Cylindrical3D].from_(obj)


# =====================================
# Spherical3D


@conversion_method(cxv.Vector, apyc.BaseRepresentation)
@conversion_method(cxv.Vector, apyc.PhysicsSphericalRepresentation)  # type: ignore[arg-type]
def sph_to_apysph(
    obj: cxv.Vector[r.Spherical3D], /
) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.Spherical3D` -> `astropy.PhysicsSphericalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Spherical3D(r=u.Q(1, "m"), theta=u.Q(2, "deg"), phi=u.Q(3, "deg"))
    >>> convert(vec, apyc.PhysicsSphericalRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, m)
        (3., 2., 1.)>

    """
    return apyc.PhysicsSphericalRepresentation(
        r=convert(obj["r"], apyu.Quantity),
        phi=convert(obj["phi"], apyu.Quantity),
        theta=convert(obj["theta"], apyu.Quantity),
    )


@conversion_method(apyc.PhysicsSphericalRepresentation, cxv.Vector)  # type: ignore[arg-type]
def apysph_to_sph(
    obj: apyc.PhysicsSphericalRepresentation, /
) -> cxv.Vector[r.Spherical3D]:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.Spherical3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.km, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> convert(sph, cx.Vector[r.Spherical3D])
    Vector(r=Distance(1., 'km'), theta=Angle(2., 'deg'), phi=Angle(3., 'deg'))

    """
    return cx.Vector[r.Spherical3D].from_(obj)


# =====================================
# LonLatSpherical3D


@conversion_method(cxv.Vector, apyc.BaseRepresentation)
@conversion_method(cxv.Vector, apyc.PhysicsSphericalRepresentation)  # type: ignore[arg-type]
def lonlatsph_to_apysph(
    obj: cxv.Vector[r.LonLatSpherical3D], /
) -> apyc.SphericalRepresentation:
    """`coordinax.LonLatSpherical3D` -> `astropy.SphericalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.LonLatSpherical3D(lon=u.Q(2, "deg"), lat=u.Q(3, "deg"),
    ...                                  distance=u.Q(1, "km"))
    >>> convert(vec, apyc.SphericalRepresentation)
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, km)
        (2., 3., 1.)>

    """
    return apyc.SphericalRepresentation(
        lon=convert(obj["lon"], apyu.Quantity),
        lat=convert(obj["lat"], apyu.Quantity),
        distance=convert(obj["distance"], apyu.Quantity),
    )


@conversion_method(apyc.SphericalRepresentation, cxv.Vector)  # type: ignore[arg-type]
def apysph_to_lonlatsph(
    obj: apyc.SphericalRepresentation, /
) -> cxv.Vector[r.LonLatSpherical3D]:
    """`astropy.SphericalRepresentation` -> `coordinax.LonLatSpherical3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=2 * u.deg, lat=3 * u.deg,
    ...                               distance=1 * u.km)
    >>> convert(sph, cx.vecs.LonLatSpherical3D)
    LonLatSpherical3D(
      lon=Angle(2., 'deg'), lat=Angle(3., 'deg'), distance=Distance(1., 'km')
    )

    """
    return cxv.Vector[r.LonLatSpherical3D].from_(obj)


# =====================================
# CartVel3D


@conversion_method(cxv.Vector, apyc.BaseDifferential)
@conversion_method(cxv.Vector, apyc.CartesianDifferential)  # type: ignore[arg-type]
def diffcart3_to_apycart3(
    obj: cxv.Vector[r.Cart3D, r.Vel], /
) -> apyc.CartesianDifferential:
    """`coordinax.CartVel3D` -> `astropy.CartesianDifferential`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dif = cx.CartVel3D.from_([1, 2, 3], "km/s")
    >>> convert(dif, apyc.CartesianDifferential)
    <CartesianDifferential (d_x, d_y, d_z) in km / s
        (1., 2., 3.)>

    """
    return apyc.CartesianDifferential(
        d_x=convert(obj["x"], apyu.Quantity),
        d_y=convert(obj["y"], apyu.Quantity),
        d_z=convert(obj["z"], apyu.Quantity),
    )


@conversion_method(apyc.CartesianDifferential, cxv.Vector)  # type: ignore[arg-type]
def apycart3_to_diffcart3(
    obj: apyc.CartesianDifferential, /
) -> cxv.Vector[r.Cart3D, r.Vel]:
    """`astropy.CartesianDifferential` -> `coordinax.CartVel3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> convert(dcart, cx.CartVel3D)
    CartVel3D(x=Q(1., 'km / s'), y=Q(2., 'km / s'), z=Q(3., 'km / s'))

    """
    return cxv.Vector[r.Cart3D, r.Vel].from_(obj)


# =====================================
# CylindricalVel


@conversion_method(cxv.Vector, apyc.BaseDifferential)
@conversion_method(cxv.Vector, apyc.CylindricalDifferential)  # type: ignore[arg-type]
def diffcyl_to_apycyl(
    obj: cxv.Vector[r.Cylindrical3D, r.Vel], /
) -> apyc.CylindricalDifferential:
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
        d_rho=convert(obj["rho"], apyu.Quantity),
        d_phi=convert(obj["phi"], apyu.Quantity),
        d_z=convert(obj["z"], apyu.Quantity),
    )


@conversion_method(apyc.CylindricalDifferential, cxv.Vector)  # type: ignore[arg-type]
def apycyl_to_diffcyl(
    obj: apyc.CylindricalDifferential, /
) -> cxv.Vector[r.Cylindrical3D, r.Vel]:
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
    return cxv.Vector[r.Cylindrical3D, r.Vel].from_(obj)


# =====================================
# SphericalVel


@conversion_method(cxv.Vector, apyc.BaseDifferential)
@conversion_method(cxv.Vector, apyc.PhysicsSphericalDifferential)  # type: ignore[arg-type]
def diffsph_to_apysph(
    obj: cxv.Vector[r.Spherical3D, r.Vel], /
) -> apyc.PhysicsSphericalDifferential:
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
        d_r=convert(obj["r"], apyu.Quantity),
        d_theta=convert(obj["theta"], apyu.Quantity),
        d_phi=convert(obj["phi"], apyu.Quantity),
    )


@conversion_method(apyc.PhysicsSphericalDifferential, cxv.Vector)  # type: ignore[arg-type]
def apysph_to_diffsph(
    obj: apyc.PhysicsSphericalDifferential, /
) -> cxv.Vector[r.Spherical3D, r.Vel]:
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
    return cxv.Vector[r.Spherical3D, r.Vel].from_(obj)


# =====================================
# LonLatSphericalVel


@conversion_method(cxv.Vector, apyc.BaseDifferential)
@conversion_method(cxv.Vector, apyc.SphericalDifferential)  # type: ignore[arg-type]
def difflonlatsph_to_apysph(
    obj: cxv.Vector[r.LonLatSpherical3D, r.Vel], /
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
        d_distance=convert(obj["distance"], apyu.Quantity),
        d_lon=convert(obj["lon"], apyu.Quantity),
        d_lat=convert(obj["lat"], apyu.Quantity),
    )


@conversion_method(apyc.SphericalDifferential, cxv.Vector)  # type: ignore[arg-type]
def apysph_to_difflonlatsph(
    obj: apyc.SphericalDifferential, /
) -> cxv.Vector[r.LonLatSpherical3D, r.Vel]:
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
    return cxv.Vector[r.LonLatSpherical3D, r.Vel].from_(obj)


# =====================================
# LonCosLatSphericalVel


@conversion_method(cxv.Vector, apyc.BaseDifferential)
@conversion_method(cxv.Vector, apyc.SphericalCosLatDifferential)  # type: ignore[arg-type]
def diffloncoslatsph_to_apysph(
    obj: cxv.Vector[r.LonCosLatSpherical3D, r.Vel], /
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
        d_distance=convert(obj["distance"], apyu.Quantity),
        d_lon_coslat=convert(obj["lon_coslat"], apyu.Quantity),
        d_lat=convert(obj["lat"], apyu.Quantity),
    )


@conversion_method(apyc.SphericalCosLatDifferential, cxv.Vector)  # type: ignore[arg-type]
def apysph_to_diffloncoslatsph(
    obj: apyc.SphericalCosLatDifferential, /
) -> cxv.Vector[r.LonCosLatSpherical3D, r.Vel]:
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
    return cxv.Vector[r.LonCosLatSpherical3D, r.Vel].from_(obj)
