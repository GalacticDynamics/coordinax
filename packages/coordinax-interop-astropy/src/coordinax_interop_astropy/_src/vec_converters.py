# mypy: disable-error-code="attr-defined"

"""Interoperability with :mod:`astropy.coordinates`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped
from typing import Any

import astropy.coordinates as apyc
import astropy.units as apyu
from plum import conversion_method, convert

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.roles as cxr

#####################################################################

# =====================================
# Quantity


@conversion_method(cx.Vector, apyu.Quantity)  # type: ignore[arg-type]  # type: ignore[arg-type]
def vec_to_q(obj: cx.Vector, /) -> Shaped[apyu.Quantity, "*batch 3"]:
    """`coordinax.AbstractPos3D` -> `astropy.units.Quantity`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> from plum import convert
    >>> import astropy.units as apyu

    >>> vec = cx.Vector.from_([1, 2, 3], "km")
    >>> convert(vec, apyu.Quantity)
    <Quantity [1., 2., 3.] km>

    >>> vec = cx.Vector.from_(
    ...     {"r": u.Q(1, "km"), "theta": u.Q(2, "deg"), "phi": u.Q(3, "deg")},
    ...     cx.charts.sph3d)
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.03485167, 0.0018265 , 0.99939084] km>

    >>> vec = cx.Vector.from_(
    ...     {"rho": u.Q(1, "km"), "phi": u.Q(2, "deg"), "z": u.Q(3, "m")},
    ...     cx.charts.cyl3d)
    >>> convert(vec, apyu.Quantity)
    <Quantity [0.99939084, 0.0348995 , 0.003     ] km>

    >>> dif = cx.Vector.from_([1, 2, 3], "km/s")
    >>> convert(dif, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s>

    >>> dif2 = cx.Vector.from_([1, 2, 3], "km/s2")
    >>> convert(dif2, AstropyQuantity)
    <Quantity [1., 2., 3.] km / s2>

    """
    return convert(convert(obj, u.Q), apyu.Quantity)


# =====================================
# Cart3D


@conversion_method(cx.Vector, apyc.CartesianRepresentation)  # type: ignore[arg-type]
def convert_vector_to_astropy(obj: cx.Vector, /) -> apyc.CartesianRepresentation:
    """`coordinax.Vector` -> `astropy.CartesianRepresentation`.

    This conversion is only used for Cartesian-charted vectors or Point-role vectors.
    For Pos/Vel/Acc roles with non-Cartesian charts, convert to the corresponding
    Astropy representation type first (e.g., Cyl → CylindricalRepresentation,
    then use Astropy's `.represent_as()` if needed).

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
    # For Cart3D, convert directly
    if obj.chart == cxc.cart3d:
        return apyc.CartesianRepresentation(
            x=convert(obj["x"], apyu.Quantity),
            y=convert(obj["y"], apyu.Quantity),
            z=convert(obj["z"], apyu.Quantity),
        )

    # For Point role, we can safely vconvert without needing a base point
    if obj.role == cxr.point:
        obj = cx.vconvert(cxc.cart3d, obj)
        return apyc.CartesianRepresentation(
            x=convert(obj["x"], apyu.Quantity),
            y=convert(obj["y"], apyu.Quantity),
            z=convert(obj["z"], apyu.Quantity),
        )

    # For Pos/Vel/Acc roles with non-Cartesian charts, we should NOT convert to
    # Cartesian here because that requires a base point. Instead, convert to the
    # corresponding Astropy representation type (which the other conversion
    # methods handle), then use Astropy's .represent_as() if you need Cartesian.
    msg = (
        f"Cannot convert {type(obj.chart).__name__} with "
        f"role {type(obj.role).__name__} directly to CartesianRepresentation. "
        f"Convert to {type(obj.chart).__name__}'s corresponding Astropy "
        "representation first, then use .represent_as() if needed."
    )
    raise TypeError(msg)


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cartesian_representation(
    cls: type[cx.Vector], obj: apyc.CartesianRepresentation, /
) -> cx.Vector:
    """Construct Vector from Astropy CartesianRepresentation.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> cx.Vector.from_(vec, cx.charts.cart3d)
    Vector(...)

    """
    return cls(
        {
            "x": convert(obj.x, u.Q),
            "y": convert(obj.y, u.Q),
            "z": convert(obj.z, u.Q),
        },
        chart=cxc.cart3d,
        role=cxr.point,
    )


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cartesian_representation_with_chart_role(
    cls: type[cx.Vector],
    obj: apyc.CartesianRepresentation,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    role: cxr.AbstractRole,
    /,
) -> cx.Vector:
    """Construct Vector from Astropy CartesianRepresentation with explicit chart/role.

    Chart and role parameters are ignored; CartesianRepresentation always maps
    to cart3d with point role.
    """
    return cls.from_(obj)


@conversion_method(apyc.CartesianRepresentation, cx.Vector)  # type: ignore[arg-type]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> cx.Vector:
    """`astropy.CartesianRepresentation` -> `coordinax.Cart3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianRepresentation

    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> convert(vec, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.cart3d)


# =====================================
# Cylindrical3D


@conversion_method(cx.Vector, apyc.BaseRepresentation)
@conversion_method(cx.Vector, apyc.CylindricalRepresentation)  # type: ignore[arg-type]
def cyl_to_apycyl(
    obj: cx.Vector[cxc.Cylindrical3D, Any, Any], /
) -> apyc.CylindricalRepresentation:
    """`Cyl3D` -> `astropy.CylindricalRepresentation`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Vector({"rho": u.Q(1, "km"), "phi": u.Q(2, "deg"),
    ...                  "z": u.Q(3, "m")}, cxc.cyl3d, cxr.point)
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cylindrical_representation(
    cls: type[cx.Vector], obj: apyc.CylindricalRepresentation, /
) -> cx.Vector:
    """Construct Vector from Astropy CylindricalRepresentation."""
    return cls.from_(
        {
            "rho": convert(obj.rho, u.Q),
            "phi": convert(obj.phi, u.Q),
            "z": convert(obj.z, u.Q),
        },
        cxc.cyl3d,
        cxr.point,
    )


@conversion_method(apyc.CylindricalRepresentation, cx.Vector)  # type: ignore[arg-type]
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> cx.Vector:
    """`astropy.CylindricalRepresentation` -> `coordinax.Cylindrical3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import CylindricalRepresentation

    >>> cyl = CylindricalRepresentation(rho=1 * u.km, phi=2 * u.deg, z=30 * u.m)
    >>> convert(cyl, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.cyl3d)


# =====================================
# Spherical3D


@conversion_method(cx.Vector, apyc.BaseRepresentation)
@conversion_method(cx.Vector, apyc.PhysicsSphericalRepresentation)  # type: ignore[arg-type]
def sph_to_apysph(
    obj: cx.Vector[cxc.Spherical3D], /
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_physics_spherical_representation(
    cls: type[cx.Vector], obj: apyc.PhysicsSphericalRepresentation, /
) -> cx.Vector:
    """Construct Vector from Astropy PhysicsSphericalRepresentation."""
    return cls.from_(
        {
            "r": convert(obj.r, u.Q),
            "theta": convert(obj.theta, u.Q),
            "phi": convert(obj.phi, u.Q),
        },
        cxc.sph3d,
        cxr.point,
    )


@conversion_method(apyc.PhysicsSphericalRepresentation, cx.Vector)  # type: ignore[arg-type]
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> cx.Vector:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.Spherical3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.km, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> convert(sph, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.sph3d)


# =====================================
# LonLatSpherical3D


@conversion_method(cx.Vector, apyc.BaseRepresentation)
@conversion_method(cx.Vector, apyc.SphericalRepresentation)  # type: ignore[arg-type]
def lonlatsph_to_apysph(
    obj: cx.Vector[cxc.LonLatSpherical3D], /
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_spherical_representation(
    cls: type[cx.Vector], obj: apyc.SphericalRepresentation, /
) -> cx.Vector:
    """Construct Vector from Astropy SphericalRepresentation."""
    return cls.from_(
        {
            "lon": convert(obj.lon, u.Q),
            "lat": convert(obj.lat, u.Q),
            "distance": convert(obj.distance, u.Q),
        },
        cxc.lonlatsph3d,
        cxr.point,
    )


@conversion_method(apyc.SphericalRepresentation, cx.Vector)  # type: ignore[arg-type]
def apysph_to_lonlatsph(obj: apyc.SphericalRepresentation, /) -> cx.Vector:
    """`astropy.SphericalRepresentation` -> `coordinax.LonLatSpherical3D`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalRepresentation

    >>> sph = SphericalRepresentation(lon=2 * u.deg, lat=3 * u.deg,
    ...                               distance=1 * u.km)
    >>> convert(sph, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.lonlatsph3d)


# =====================================
# CartVel3D


@conversion_method(cx.Vector, apyc.BaseDifferential)
@conversion_method(cx.Vector, apyc.CartesianDifferential)  # type: ignore[arg-type]
def diffcart3_to_apycart3(
    obj: cx.Vector[cxc.Cart3D, cxr.PhysVel], /
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cartesian_differential(
    cls: type[cx.Vector], obj: apyc.CartesianDifferential, /
) -> cx.Vector:
    """Construct Vector from Astropy CartesianDifferential."""
    return cls(
        {
            "x": convert(obj.d_x, u.Q),
            "y": convert(obj.d_y, u.Q),
            "z": convert(obj.d_z, u.Q),
        },
        cxc.cart3d,
        cxr.phys_vel,
    )


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cartesian_differential_with_chart_role(
    cls: type[cx.Vector],
    obj: apyc.CartesianDifferential,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    role: cxr.AbstractRole,
    /,
) -> cx.Vector:
    """Construct Vector from Astropy CartesianDifferential with explicit chart/role.

    Chart and role parameters are ignored; CartesianDifferential always maps
    to cart3d with vel role.
    """
    return cls.from_(obj)


@conversion_method(apyc.CartesianDifferential, cx.Vector)  # type: ignore[arg-type]
def apycart3_to_diffcart3(obj: apyc.CartesianDifferential, /) -> cx.Vector:
    """`astropy.CartesianDifferential` -> `coordinax.CartVel3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from astropy.coordinates import CartesianDifferential

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> convert(dcart, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.cart3d, cxr.phys_vel)


# =====================================
# CylindricalVel


@conversion_method(cx.Vector, apyc.BaseDifferential)
@conversion_method(cx.Vector, apyc.CylindricalDifferential)  # type: ignore[arg-type]
def diffcyl_to_apycyl(
    obj: cx.Vector[cxc.Cylindrical3D, cxr.PhysVel], /
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_cylindrical_differential(
    cls: type[cx.Vector], obj: apyc.CylindricalDifferential, /
) -> cx.Vector:
    """Construct Vector from Astropy CylindricalDifferential."""
    return cls(
        {
            "rho": convert(obj.d_rho, u.Q),
            "phi": convert(obj.d_phi, u.Q),
            "z": convert(obj.d_z, u.Q),
        },
        chart=cxc.cyl3d,
        role=cxr.phys_vel,
    )


@conversion_method(apyc.CylindricalDifferential, cx.Vector)  # type: ignore[arg-type]
def apycyl_to_diffcyl(obj: apyc.CylindricalDifferential, /) -> cx.Vector:
    """`astropy.CylindricalDifferential` -> `coordinax.CylindricalVel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> import coordinax as cx

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> convert(dcyl, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.cyl3d, cxr.phys_vel)


# =====================================
# SphericalVel


@conversion_method(cx.Vector, apyc.BaseDifferential)
@conversion_method(cx.Vector, apyc.PhysicsSphericalDifferential)  # type: ignore[arg-type]
def diffsph_to_apysph(
    obj: cx.Vector[cxc.Spherical3D, cxr.PhysVel], /
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_physics_spherical_differential(
    cls: type[cx.Vector], obj: apyc.PhysicsSphericalDifferential, /
) -> cx.Vector:
    """Construct Vector from Astropy PhysicsSphericalDifferential."""
    return cls(
        {
            "r": convert(obj.d_r, u.Q),
            "theta": convert(obj.d_theta, u.Q),
            "phi": convert(obj.d_phi, u.Q),
        },
        chart=cxc.sph3d,
        role=cxr.phys_vel,
    )


@conversion_method(apyc.PhysicsSphericalDifferential, cx.Vector)  # type: ignore[arg-type]
def apysph_to_diffsph(obj: apyc.PhysicsSphericalDifferential, /) -> cx.Vector:
    """`astropy.PhysicsSphericalDifferential` -> SphericalVel.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import PhysicsSphericalDifferential

    >>> dif = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                    d_phi=3 * u.mas/u.yr)
    >>> convert(dif, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.sph3d, cxr.phys_vel)


# =====================================
# LonLatSphericalVel


@conversion_method(cx.Vector, apyc.BaseDifferential)
@conversion_method(cx.Vector, apyc.SphericalDifferential)  # type: ignore[arg-type]
def difflonlatsph_to_apysph(
    obj: cx.Vector[cxc.LonLatSpherical3D, cxr.PhysVel], /
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_spherical_differential(
    cls: type[cx.Vector], obj: apyc.SphericalDifferential, /
) -> cx.Vector:
    """Construct Vector from Astropy SphericalDifferential."""
    return cls(
        {
            "lon": convert(obj.d_lon, u.Q),
            "lat": convert(obj.d_lat, u.Q),
            "distance": convert(obj.d_distance, u.Q),
        },
        chart=cxc.lonlatsph3d,
        role=cxr.phys_vel,
    )


@conversion_method(apyc.SphericalDifferential, cx.Vector)  # type: ignore[arg-type]
def apysph_to_difflonlatsph(obj: apyc.SphericalDifferential, /) -> cx.Vector:
    """`astropy.coordinates.SphericalDifferential` -> LonLatSphericalVel.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalDifferential

    >>> dif = SphericalDifferential(d_distance=1 * u.km / u.s, d_lat=2 * u.mas/u.yr,
    ...                             d_lon=3 * u.mas/u.yr)
    >>> convert(dif, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.lonlatsph3d, cxr.phys_vel)


# =====================================
# LonCosLatSphericalVel


@conversion_method(cx.Vector, apyc.BaseDifferential)
@conversion_method(cx.Vector, apyc.SphericalCosLatDifferential)  # type: ignore[arg-type]
def diffloncoslatsph_to_apysph(
    obj: cx.Vector[cxc.LonCosLatSpherical3D, cxr.PhysVel], /
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


@cx.Vector.from_.dispatch  # type: ignore[untyped-decorator]
def from_astropy_spherical_coslat_differential(
    cls: type[cx.Vector], obj: apyc.SphericalCosLatDifferential, /
) -> cx.Vector:
    """Construct Vector from Astropy SphericalCosLatDifferential."""
    return cls(
        {
            "lon_coslat": convert(obj.d_lon_coslat, u.Q),
            "lat": convert(obj.d_lat, u.Q),
            "distance": convert(obj.d_distance, u.Q),
        },
        chart=cxc.loncoslatsph3d,
        role=cxr.phys_vel,
    )


@conversion_method(apyc.SphericalCosLatDifferential, cx.Vector)  # type: ignore[arg-type]
def apysph_to_diffloncoslatsph(obj: apyc.SphericalCosLatDifferential, /) -> cx.Vector:
    """`astropy.SphericalCosLatDifferential` -> LonCosLatSphericalVel.

    Examples
    --------
    >>> import astropy.units as u
    >>> import coordinax as cx
    >>> from astropy.coordinates import SphericalCosLatDifferential

    >>> dif = SphericalCosLatDifferential(d_distance=1 * u.km / u.s,
    ...                                   d_lat=2 * u.mas/u.yr,
    ...                                   d_lon_coslat=3 * u.mas/u.yr)
    >>> convert(dif, cx.Vector)
    Vector(...)

    """
    return cx.Vector.from_(obj, cxc.loncoslatsph3d, cxr.phys_vel)
