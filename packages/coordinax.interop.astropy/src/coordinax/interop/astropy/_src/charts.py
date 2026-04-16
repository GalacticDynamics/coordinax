"""Astropy Charts compatibility."""

__all__: tuple[str, ...] = ()


import plum

import astropy.coordinates as apyc

import coordinax.charts as cxc


@plum.dispatch
def guess_chart(obj: apyc.RadialRepresentation) -> cxc.Radial1D:
    """Guess `astropy.coordinates.RadialRepresentation` -> ``Radial1D``.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.RadialRepresentation(distance=1 * apyu.kpc)
    >>> cxc.guess_chart(vec)
    Radial1D()

    """
    return cxc.radial1d


# ===================================================================


@plum.dispatch
def guess_chart(obj: apyc.UnitSphericalRepresentation) -> cxc.LonLatSphericalTwoSphere:
    """Guess ``astropy.UnitSphericalRepresentation`` -> ``LonLatSphericalTwoSphere``.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.UnitSphericalRepresentation(lon=90 * apyu.deg, lat=45 * apyu.deg)
    >>> cxc.guess_chart(vec)
    LonLatSphericalTwoSphere()

    """
    return cxc.lonlat_sph2


# ===================================================================


@plum.dispatch
def guess_chart(obj: apyc.CartesianRepresentation) -> cxc.Cart3D:
    """Guess `astropy.coordinates.CartesianRepresentation` -> ``Cart3D``.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.charts as cxc

    >>> cxc.guess_chart(apyc.CartesianRepresentation(1, 2, 3))
    Cart3D()

    """
    return cxc.cart3d


@plum.dispatch
def guess_chart(obj: apyc.CylindricalRepresentation) -> cxc.Cylindrical3D:
    """Guess `astropy.coordinates.CylindricalRepresentation` -> ``Cylindrical3D``.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.CylindricalRepresentation(1 * apyu.km, 2 * apyu.deg, 3 * apyu.km)
    >>> cxc.guess_chart(vec)
    Cylindrical3D()

    """
    return cxc.cyl3d


@plum.dispatch
def guess_chart(obj: apyc.PhysicsSphericalRepresentation) -> cxc.Spherical3D:
    """Guess `astropy.coordinates.PhysicsSphericalRepresentation` -> ``Spherical3D``.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.PhysicsSphericalRepresentation(
    ...     r=1 * apyu.kpc, theta=45 * apyu.deg, phi=90 * apyu.deg)
    >>> cxc.guess_chart(vec)
    Spherical3D()

    """
    return cxc.sph3d


@plum.dispatch
def guess_chart(obj: apyc.SphericalRepresentation) -> cxc.LonLatSpherical3D:
    """Guess `astropy.coordinates.SphericalRepresentation` -> ``LonLatSpherical3D``.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.SphericalRepresentation(
    ...     lon=90 * apyu.deg, lat=45 * apyu.deg, distance=1 * apyu.kpc)
    >>> cxc.guess_chart(vec)
    LonLatSpherical3D()

    """
    return cxc.lonlat_sph3d
