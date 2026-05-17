# mypy: disable-error-code="attr-defined"

"""Interoperability with {mod}`astropy.coordinates`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped
from typing import Any, TypeVar, cast

import plum

import astropy.coordinates as apyc
import astropy.units as apyu
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.vectors as cxv

V = TypeVar("V", bound=cxv.AbstractVector)


def check_semantics(obj: V, /, need: type[cxr.AbstractSemanticKind]) -> V:
    """Check that the vector has the needed role, and raise ValueError if not."""
    if not isinstance(obj.rep.semantic_kind, need):
        msg = (
            f"Expected role {need.__name__}, "
            f"but got {type(obj.rep.semantic_kind).__name__}."
        )
        raise TypeError(msg)
    return obj


#####################################################################
# Quantity


@plum.conversion_method(cxv.Point, apyu.Quantity)
def vec_to_q(obj: cxv.Point, /) -> Shaped[apyu.Quantity, "*batch 3"]:
    """`coordinax.Point` -> `astropy.units.Quantity`.

    >>> import unxt as u
    >>> import coordinax.vectors as cxv
    >>> from plum import convert
    >>> import astropy.units as apyu

    >>> point = cxv.Point.from_([1, 2, 3], "km")
    >>> convert(point, apyu.Quantity)
    <Quantity [1., 2., 3.] km>

    >>> point = cxv.Point.from_(
    ...     {"r": u.Q(1, "km"), "theta": u.Q(2, "deg"), "phi": u.Q(3, "deg")},
    ...     cxc.sph3d)
    >>> convert(point, apyu.Quantity)
    <Quantity (1, 2, 3) (km, deg, deg)>

    >>> point = cxv.Point.from_(
    ...     {"rho": u.Q(1, "km"), "phi": u.Q(2, "deg"), "z": u.Q(3, "m")},
    ...     cxc.cyl3d)
    >>> convert(point, apyu.Quantity)
    <Quantity (1, 2, 3) (km, deg, m)>

    """
    return plum.convert(plum.convert(obj, u.AbstractQuantity), apyu.Quantity)


#####################################################################
# Astropy Representations


@plum.conversion_method(cxv.Point, apyc.BaseRepresentation)
def convert_vector_to_astropy(obj: cxv.Point, /) -> apyc.BaseRepresentation:
    """Convert a `coordinax.Point` to a `astropy.coordinates.BaseRepresentation`.

    The specific Astropy representation type (e.g., Cartesian vs. Cylindrical)
    is determined by the chart of the input point. The point's role must be
    compatible with a position/location (e.g., Point), and the chart must be one
    of the supported types (e.g., Cart3D, Cyl3D, Sph3D, etc.), or else a
    ValueError will be raised.

    >>> import unxt as u
    >>> import coordinax.vectors as cxv
    >>> from plum import convert

    >>> point = cxv.Point.from_([1, 2, 3], "km")
    >>> convert(point, apyc.BaseRepresentation)
    <CartesianRepresentation (x, y, z) in km
        (1., 2., 3.)>

    >>> point = cxv.Point.from_(
    ...     {"r": u.Q(1, "km"), "theta": u.Q(2, "deg"), "phi": u.Q(3, "deg")},
    ...     cxc.sph3d)
    >>> convert(point, apyc.BaseRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, km)
        (3., 2., 1.)>

    >>> point = cxv.Point.from_(
    ...     {"rho": u.Q(1, "km"), "phi": u.Q(2, "deg"), "z": u.Q(3, "m")},
    ...     cxc.cyl3d)
    >>> convert(point, apyc.BaseRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (km, deg, m)
        (1., 2., 3.)>

    """
    # Check that the vector has a compatible role, and raise ValueError if not.
    obj = check_semantics(obj, need=cxr.Location)

    # Dispatch to specific conversion method based on chart.
    if obj.chart == cxc.cart3d:
        return cart3_to_apycart3(obj)
    if obj.chart == cxc.cyl3d:
        return cyl_to_apycyl(obj)
    if obj.chart == cxc.sph3d:
        return sph_to_apysph(obj)
    if obj.chart == cxc.lonlat_sph3d:
        return lonlatsph_to_apysph(obj)

    msg = f"Cannot convert vector with chart {obj.chart} to an Astropy representation."
    raise ValueError(msg)


# =====================================
# Cart3D


@plum.conversion_method(cxv.Point, apyc.CartesianRepresentation)
def cart3_to_apycart3(obj: cxv.Point, /) -> apyc.CartesianRepresentation:
    """`coordinax.Point` -> `astropy.CartesianRepresentation`.

    This conversion is only used for Cartesian-charted points or Point-role
    points.  For Pos/Vel/Acc roles with non-Cartesian charts, convert to the
    corresponding Astropy representation type first (e.g., Cyl →
    CylindricalRepresentation, then use Astropy's `.represent_as()` if needed).

    >>> import coordinax.vectors as cxv
    >>> point = cxv.Point.from_([1, 2, 3], "km")
    >>> convert(point, apyc.CartesianRepresentation)
    <CartesianRepresentation (x, y, z) in km
        (1., 2., 3.)>

    >>> convert(point, apyc.BaseRepresentation)
    <CartesianRepresentation (x, y, z) in km
        (1., 2., 3.)>

    """
    # Check that the vector has a compatible role, and raise ValueError if not.
    obj = check_semantics(obj, need=cxr.Location)
    # Convert to Cartesian if not already in that chart.
    if obj.chart != cxc.cart3d:
        obj = cast("cxv.Point", obj.cconvert(cxc.cart3d))

    return apyc.CartesianRepresentation(
        x=plum.convert(obj["x"], apyu.Quantity),
        y=plum.convert(obj["y"], apyu.Quantity),
        z=plum.convert(obj["z"], apyu.Quantity),
    )


@plum.conversion_method(apyc.CartesianRepresentation, cxv.Point)
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> cxv.Point:
    """`astropy.CartesianRepresentation` -> `coordinax.Cart3D`.

    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import CartesianRepresentation
    >>> vec = CartesianRepresentation(1, 2, 3, unit="km")
    >>> print(convert(vec, cxv.Point))
    <Point: chart=Cart3D (x, y, z) [km]
        [1. 2. 3.]>

    """
    return cxv.Point.from_(obj)  # ty: ignore[invalid-return-type]


# =====================================
# Cylindrical3D


@plum.conversion_method(cxv.Point, apyc.CylindricalRepresentation)
def cyl_to_apycyl(
    obj: cxv.Point[cxc.Cylindrical3D, Any], /
) -> apyc.CylindricalRepresentation:
    """`Cyl3D` -> `astropy.CylindricalRepresentation`.

    >>> import unxt as u
    >>> import coordinax.vectors as cxv

    >>> vec = cxv.Point.from_({"rho": u.Q(1, "km"), "phi": u.Q(2, "deg"),
    ...                        "z": u.Q(3, "m")}, cxc.cyl3d)
    >>> convert(vec, apyc.CylindricalRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (km, deg, m)
        (1., 2., 3.)>

    >>> convert(vec, apyc.BaseRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (km, deg, m)
        (1., 2., 3.)>

    """
    # Check that the vector has a compatible role, and raise ValueError if not.
    obj = check_semantics(obj, need=cxr.Location)
    # Convert to CylindricalRepresentation if not already in that chart.
    if obj.chart != cxc.cyl3d:
        obj: cxv.Point = obj.cconvert(cxc.cyl3d)
    # Now safely convert components.
    return apyc.CylindricalRepresentation(
        rho=plum.convert(obj["rho"], apyu.Quantity),
        phi=plum.convert(obj["phi"], apyu.Quantity),
        z=plum.convert(obj["z"], apyu.Quantity),
    )


@plum.conversion_method(apyc.CylindricalRepresentation, cxv.Point)
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> cxv.Point:
    """`astropy.CylindricalRepresentation` -> `coordinax.Cylindrical3D`.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import coordinax.vectors as cxv

    >>> cyl = apyc.CylindricalRepresentation(rho=1 * apyu.km, phi=2 * apyu.deg,
    ...                                      z=30 * apyu.m)
    >>> print(convert(cyl, cxv.Point))
    <Point: chart=Cylindrical3D (rho[km], phi[deg], z[m])
        [ 1.  2. 30.]>

    """
    return cast("cxv.Point", cxv.Point.from_(obj))


# =====================================
# Spherical3D


@plum.conversion_method(cxv.Point, apyc.PhysicsSphericalRepresentation)
def sph_to_apysph(
    obj: cxv.Point[cxc.Spherical3D], /
) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.Point` -> `astropy.PhysicsSphericalRepresentation`.

    >>> import unxt as u
    >>> import coordinax.vectors as cxv
    >>> vec = cxv.Point.from_({"r": u.Q(1,"m"), "theta": u.Q(2,"deg"),
    ...                         "phi": u.Q(3,"deg")}, cxc.sph3d)
    >>> convert(vec, apyc.PhysicsSphericalRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, m)
        (3., 2., 1.)>

    """
    # Check that the vector has a compatible role, and raise ValueError if not.
    obj = check_semantics(obj, need=cxr.Location)
    # Convert to SphericalRepresentation if not already in that chart.
    if obj.chart != cxc.sph3d:
        obj: cxv.Point = obj.cconvert(cxc.sph3d)
    # Now safely convert components.
    return apyc.PhysicsSphericalRepresentation(
        r=plum.convert(obj["r"], apyu.Quantity),
        phi=plum.convert(obj["phi"], apyu.Quantity),
        theta=plum.convert(obj["theta"], apyu.Quantity),
    )


@plum.conversion_method(apyc.PhysicsSphericalRepresentation, cxv.Point)
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> cxv.Point:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.Spherical3D`.

    >>> import astropy.units as u
    >>> import coordinax.vectors as cxv
    >>> from astropy.coordinates import PhysicsSphericalRepresentation

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.km, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> print(convert(sph, cxv.Point))
    <Point: chart=Spherical3D (r[km], theta[deg], phi[deg])
        [1. 2. 3.]>

    """
    return cxv.Point.from_(obj)  # ty: ignore[invalid-return-type]


# =====================================
# LonLatSpherical3D


@plum.conversion_method(cxv.Point, apyc.SphericalRepresentation)
def lonlatsph_to_apysph(
    obj: cxv.Point[cxc.LonLatSpherical3D, Any], /
) -> apyc.SphericalRepresentation:
    """`coordinax.LonLatSpherical3D` -> `astropy.SphericalRepresentation`.

    >>> import unxt as u
    >>> import coordinax.vectors as cxv

    >>> vec = cxv.Point.from_({"lon": u.Q(1, "deg"), "lat": u.Q(2, "deg"),
    ...                         "distance": u.Q(3, "km")}, cxc.lonlat_sph3d)
    >>> convert(vec, apyc.SphericalRepresentation)
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, km)
        (1., 2., 3.)>

    """
    # Check that the vector has a compatible role, and raise TypeError if not.
    obj = check_semantics(obj, need=cxr.Location)
    # Convert to SphericalRepresentation if not already in that chart.
    if obj.chart != cxc.lonlat_sph3d:
        obj: cxv.Point = obj.cconvert(cxc.lonlat_sph3d)
    # Now safely convert components.
    return apyc.SphericalRepresentation(
        lon=plum.convert(obj["lon"], apyu.Quantity),
        lat=plum.convert(obj["lat"], apyu.Quantity),
        distance=plum.convert(obj["distance"], apyu.Quantity),
    )


@plum.conversion_method(apyc.SphericalRepresentation, cxv.Point)
def apysph_to_lonlatsph(obj: apyc.SphericalRepresentation, /) -> cxv.Point:
    """`astropy.SphericalRepresentation` -> `coordinax.LonLatSpherical3D`.

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc
    >>> import coordinax.vectors as cxv

    >>> sph = apyc.SphericalRepresentation(lon=2 * apyu.deg, lat=3 * apyu.deg,
    ...                                    distance=1 * apyu.km)
    >>> print(convert(sph, cxv.Point))
    <Point: chart=LonLatSpherical3D (lon[deg], lat[deg], distance[km])
        [2. 3. 1.]>

    """
    return cxv.Point.from_(obj)  # ty: ignore[invalid-return-type]
