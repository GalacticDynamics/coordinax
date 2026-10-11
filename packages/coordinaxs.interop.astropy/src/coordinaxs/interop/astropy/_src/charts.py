"""Astropy Charts compatibility."""

__all__: tuple[str, ...] = ()


import astropy.coordinates as apyc
import astropy.units as apyu
import plum

import unxt as u

import coordinax.charts as cxc
import coordinaxs.api.charts as cxcapi
from coordinaxs.api.custom_types import CDict, CKeys


@plum.dispatch
def guess_chart(obj: apyc.RadialRepresentation) -> cxc.Radial1D:
    """Guess `astropy.coordinates.RadialRepresentation` -> ``Radial1D``.

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> vec = apyc.RadialRepresentation(distance=1 * apyu.kpc)
    >>> cxc.guess_chart(vec)
    Radial1D(M=Rn(1))

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
    LonLatSphericalTwoSphere(M=Sn(2))

    """
    return cxc.lonlat_sph2


# ===================================================================


@plum.dispatch
def guess_chart(obj: apyc.CartesianRepresentation) -> cxc.Cart3D:
    """Guess `astropy.coordinates.CartesianRepresentation` -> ``Cart3D``.

    >>> import astropy.coordinates as apyc
    >>> import coordinax.charts as cxc

    >>> cxc.guess_chart(apyc.CartesianRepresentation(1, 2, 3))
    Cart3D(M=Rn(3))

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
    Cylindrical3D(M=Rn(3))

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
    Spherical3D(M=Rn(3))

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
    LonLatSpherical3D(M=Rn(3))

    """
    return cxc.lonlat_sph3d


# ===================================================================
# Astropy Quantity -> CDict
#
# An astropy `Quantity` is an `numpy.ndarray` subclass, so without these it
# matches the bare `ArrayLike` methods, which drop its unit (#1024). Each one
# converts to `unxt.Q` and re-dispatches, so it gets exactly the `unxt` result,
# including the homogeneous-dimension check of ``cdict(Q, chart)``.


@plum.dispatch
def cdict(obj: apyu.Quantity, /) -> CDict:
    """Split an astropy ``Quantity`` into a component dict, keeping its unit.

    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> cxc.cdict(apyu.Quantity([1, 2, 3], "kpc"))
    {'x': Q(1., 'kpc'), 'y': Q(2., 'kpc'), 'z': Q(3., 'kpc')}

    """
    return cxcapi.cdict(plum.convert(obj, u.Q))  # ty: ignore[invalid-return-type]


@plum.dispatch
def cdict(obj: apyu.Quantity, keys: CKeys, /) -> CDict:
    """Split an astropy ``Quantity`` along the given keys, keeping its unit.

    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> cxc.cdict(apyu.Quantity([1, 2], "m"), ("a", "b"))
    {'a': Q(1., 'm'), 'b': Q(2., 'm')}

    """
    return cxcapi.cdict(plum.convert(obj, u.Q), keys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def cdict(obj: apyu.Quantity, chart: cxc.AbstractChart, /) -> CDict:  # type: ignore[type-arg]
    """Split an astropy ``Quantity`` across a chart, keeping its unit.

    >>> import astropy.units as apyu
    >>> import coordinax.charts as cxc

    >>> cxc.cdict(apyu.Quantity([1, 2, 3], "kpc"), cxc.cart3d)
    {'x': Q(1., 'kpc'), 'y': Q(2., 'kpc'), 'z': Q(3., 'kpc')}

    """
    return cxcapi.cdict(plum.convert(obj, u.Q), chart)  # ty: ignore[invalid-return-type]
