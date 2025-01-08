"""Representation of coordinates in different systems."""
# ruff: noqa: N803, N806

__all__: list[str] = []


from plum import conversion_method

from .base import AbstractPos3D
from .cartesian import CartesianPos3D
from .cylindrical import CylindricalPos
from .lonlatspherical import LonLatSphericalPos
from .mathspherical import MathSphericalPos
from .spherical import SphericalPos
from coordinax._src.vectors.api import vconvert


@conversion_method(type_from=AbstractPos3D, type_to=CartesianPos3D)  # type: ignore[type-abstract,arg-type]
def convert_pos3d_to_cart3d(pos: AbstractPos3D) -> CartesianPos3D:
    """Convert a 3D position vector to Cartesian coordinates.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> print(convert(q, cx.vecs.CartesianPos3D))
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [1 2 3]>

    >>> q = cx.vecs.CylindricalPos(rho=u.Quantity(1, "kpc"),
    ...                            phi=u.Quantity(0, "deg"),
    ...                            z=u.Quantity(3, "kpc"))
    >>> print(convert(q, cx.vecs.CartesianPos3D))
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [1. 0. 3.]>

    """
    return vconvert(CartesianPos3D, pos)


@conversion_method(type_from=AbstractPos3D, type_to=CylindricalPos)  # type: ignore[type-abstract,arg-type]
def convert_cart3d_to_cylindrical(pos: AbstractPos3D) -> CylindricalPos:
    """Convert a 3D position vector to cylindrical coordinates.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> print(convert(q, cx.vecs.CylindricalPos))
    <CylindricalPos (rho[kpc], phi[rad], z[kpc])
        [2.236 1.107 3.   ]>

    >>> q = cx.vecs.SphericalPos(r=u.Quantity(1, "kpc"),
    ...                          theta=u.Quantity(0, "deg"),
    ...                          phi=u.Quantity(0, "deg"))
    >>> print(convert(q, cx.vecs.CylindricalPos))
    <CylindricalPos (rho[kpc], phi[deg], z[kpc])
        [0. 0. 1.]>

    """
    return vconvert(CylindricalPos, pos)


@conversion_method(type_from=AbstractPos3D, type_to=SphericalPos)  # type: ignore[type-abstract,arg-type]
def convert_cylindrical_to_spherical(pos: AbstractPos3D) -> SphericalPos:
    """Convert a 3D position vector to spherical coordinates.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> print(convert(q, cx.vecs.SphericalPos))
    <SphericalPos (r[kpc], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    >>> q = cx.vecs.CylindricalPos(rho=u.Quantity(1, "kpc"),
    ...                            phi=u.Quantity(0, "deg"),
    ...                            z=u.Quantity(3, "kpc"))
    >>> print(convert(q, cx.vecs.SphericalPos))
    <SphericalPos (r[kpc], theta[rad], phi[deg])
        [3.162 0.322 0.   ]>

    """
    return vconvert(SphericalPos, pos)


@conversion_method(type_from=AbstractPos3D, type_to=MathSphericalPos)  # type: ignore[type-abstract,arg-type]
def convert_spherical_to_math_spherical(pos: AbstractPos3D) -> MathSphericalPos:
    """Convert a 3D position vector to mathematical spherical coordinates.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> print(convert(q, cx.vecs.MathSphericalPos))
    <MathSphericalPos (r[kpc], theta[rad], phi[rad])
        [3.742 1.107 0.641]>

    >>> q = cx.vecs.SphericalPos(r=u.Quantity(1, "kpc"),
    ...                          theta=u.Quantity(0, "deg"),
    ...                          phi=u.Quantity(0, "deg"))
    >>> print(convert(q, cx.vecs.MathSphericalPos))
    <MathSphericalPos (r[kpc], theta[deg], phi[deg])
        [1 0 0]>

    """
    return vconvert(MathSphericalPos, pos)


@conversion_method(type_from=AbstractPos3D, type_to=LonLatSphericalPos)  # type: ignore[arg-type,type-abstract]
def convert_math_spherical_to_lonlat_spherical(
    pos: AbstractPos3D,
) -> LonLatSphericalPos:
    """Convert a 3D position vector to lon-lat spherical coordinates.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> print(convert(q, cx.vecs.LonLatSphericalPos))
    <LonLatSphericalPos (lon[rad], lat[deg], distance[kpc])
        [ 1.107 53.301  3.742]>

    >>> q = cx.vecs.MathSphericalPos(r=u.Quantity(1, "kpc"),
    ...                              theta=u.Quantity(0, "deg"),
    ...                              phi=u.Quantity(0, "deg"))
    >>> print(convert(q, cx.vecs.LonLatSphericalPos))
    <LonLatSphericalPos (lon[rad], lat[deg], distance[kpc])
        [ 0. 90.  1.]>

    """
    return vconvert(LonLatSphericalPos, pos)
