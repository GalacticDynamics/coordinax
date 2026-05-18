"""Astropy Angle compatibility."""

__all__: tuple[str, ...] = (
    "convert_cx_cdict_to_astropy_cartrep",
    "convert_cx_cdict_to_astropy_cylrep",
    "convert_cx_cdict_to_astropy_physsphrep",
    "convert_cx_cdict_to_astropy_sphrep",
)


import plum

import astropy.coordinates as apyc
import astropy.units as apyu
import unxt as u

import coordinax.charts as cxc
from .custom_types import CDict


@plum.conversion_method(type_from=dict, type_to=apyc.CartesianRepresentation)
def convert_cx_cdict_to_astropy_cartrep(p: CDict, /) -> apyc.CartesianRepresentation:
    """Convert a CDict to an astropy CartesianRepresentation.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import plum

    >>> p = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> plum.convert(p, apyc.CartesianRepresentation)
    <CartesianRepresentation (x, y, z) in m
        (1., 2., 3.)>

    """
    cxc.cart3d.check_data(p, keys=True)
    return apyc.CartesianRepresentation(
        x=plum.convert(p["x"], apyu.Quantity),
        y=plum.convert(p["y"], apyu.Quantity),
        z=plum.convert(p["z"], apyu.Quantity),
    )


@plum.dispatch
def cdict(r: apyc.CartesianRepresentation) -> CDict:
    """Convert an astropy CartesianRepresentation to a CDict.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.CartesianRepresentation(1 * apyu.km, 2 * apyu.km, 3 * apyu.km)
    >>> cxc.cdict(vec)
    {'x': Q(1., 'km'), 'y': Q(2., 'km'), 'z': Q(3., 'km')}

    """
    return {
        "x": plum.convert(r.x, u.Q),  # ty: ignore[unresolved-attribute]
        "y": plum.convert(r.y, u.Q),  # ty: ignore[unresolved-attribute]
        "z": plum.convert(r.z, u.Q),  # ty: ignore[unresolved-attribute]
    }


# ===================================================================


@plum.conversion_method(type_from=dict, type_to=apyc.CylindricalRepresentation)
def convert_cx_cdict_to_astropy_cylrep(p: CDict, /) -> apyc.CylindricalRepresentation:
    """Convert a CDict to an astropy CylindricalRepresentation.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import plum

    >>> p = {"rho": u.Q(1, "m"), "phi": u.Q(45, "deg"), "z": u.Q(3, "m")}
    >>> plum.convert(p, apyc.CylindricalRepresentation)
    <CylindricalRepresentation (rho, phi, z) in (m, deg, m)
        (1., 45., 3.)>

    """
    cxc.cyl3d.check_data(p, keys=True)
    return apyc.CylindricalRepresentation(
        rho=plum.convert(p["rho"], apyu.Quantity),
        phi=plum.convert(p["phi"], apyu.Quantity),
        z=plum.convert(p["z"], apyu.Quantity),
    )


@plum.dispatch
def cdict(r: apyc.CylindricalRepresentation) -> CDict:
    """Convert an astropy CylindricalRepresentation to a CDict.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.CylindricalRepresentation(1 * apyu.m, 45 * apyu.deg, 3 * apyu.m)
    >>> cxc.cdict(vec)
    {'rho': Q(1., 'm'), 'phi': Q(45., 'deg'), 'z': Q(3., 'm')}

    """
    return {
        "rho": plum.convert(r.rho, u.Q),
        "phi": plum.convert(r.phi, u.Q),
        "z": plum.convert(r.z, u.Q),
    }


# ===================================================================


@plum.conversion_method(type_from=dict, type_to=apyc.PhysicsSphericalRepresentation)
def convert_cx_cdict_to_astropy_physsphrep(
    p: CDict, /
) -> apyc.PhysicsSphericalRepresentation:
    """Convert a CDict to an astropy PhysicsSphericalRepresentation.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import plum

    >>> p = {"phi": u.Q(45, "deg"), "theta": u.Q(30, "deg"), "r": u.Q(1, "m")}
    >>> plum.convert(p, apyc.PhysicsSphericalRepresentation)
    <PhysicsSphericalRepresentation (phi, theta, r) in (deg, deg, m)
        (45., 30., 1.)>

    """
    cxc.sph3d.check_data(p, keys=True)
    return apyc.PhysicsSphericalRepresentation(
        phi=plum.convert(p["phi"], apyu.Quantity),
        theta=plum.convert(p["theta"], apyu.Quantity),
        r=plum.convert(p["r"], apyu.Quantity),
    )


@plum.dispatch
def cdict(r: apyc.PhysicsSphericalRepresentation) -> CDict:
    """Convert an astropy PhysicsSphericalRepresentation to a CDict.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.PhysicsSphericalRepresentation(
    ...     phi=45 * apyu.deg, theta=30 * apyu.deg, r=1 * apyu.m)
    >>> cxc.cdict(vec)
    {'r': Q(1., 'm'), 'theta': Q(30., 'deg'), 'phi': Q(45., 'deg')}

    """
    return {
        "r": plum.convert(r.r, u.Q),
        "theta": plum.convert(r.theta, u.Q),
        "phi": plum.convert(r.phi, u.Q),
    }


# ===================================================================


@plum.conversion_method(type_from=dict, type_to=apyc.SphericalRepresentation)
def convert_cx_cdict_to_astropy_sphrep(p: CDict, /) -> apyc.SphericalRepresentation:
    """Convert a CDict to an astropy SphericalRepresentation.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import plum

    >>> p = {"lon": u.Q(45, "deg"), "lat": u.Q(30, "deg"), "distance": u.Q(1, "m")}
    >>> plum.convert(p, apyc.SphericalRepresentation)
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, m)
        (45., 30., 1.)>

    """
    cxc.lonlat_sph3d.check_data(p, keys=True)
    return apyc.SphericalRepresentation(
        lon=plum.convert(p["lon"], apyu.Quantity),
        lat=plum.convert(p["lat"], apyu.Quantity),
        distance=plum.convert(p["distance"], apyu.Quantity),
    )


@plum.dispatch
def cdict(r: apyc.SphericalRepresentation) -> CDict:
    """Convert an astropy SphericalRepresentation to a CDict.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.SphericalRepresentation(
    ...     lon=90 * apyu.deg, lat=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> cxc.cdict(vec)
    {'lon': Q(90., 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'kpc')}

    """
    return {
        "lon": plum.convert(r.lon, u.Q),
        "lat": plum.convert(r.lat, u.Q),
        "distance": plum.convert(r.distance, u.Q),
    }


# ===================================================================
