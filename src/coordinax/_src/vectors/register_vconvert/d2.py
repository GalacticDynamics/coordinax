"""Transformations between representations."""

__all__: list[str] = []

from typing import Any
from warnings import warn

from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import AbstractQuantity, Quantity

from coordinax._src.vectors.d1 import CartesianPos1D, RadialPos
from coordinax._src.vectors.d2 import AbstractPos2D, CartesianPos2D, PolarPos
from coordinax._src.vectors.d3 import (
    AbstractPos3D,
    CartesianPos3D,
    CylindricalPos,
    MathSphericalPos,
    SphericalPos,
)
from coordinax._src.vectors.exceptions import IrreversibleDimensionChange


@dispatch.multi(
    (type[CylindricalPos], CartesianPos2D),
    (type[SphericalPos], CartesianPos2D),
    (type[MathSphericalPos], CartesianPos2D),
)
def vconvert(
    target: type[AbstractPos3D],
    current: AbstractPos2D,
    /,
    z: AbstractQuantity = Quantity(0, "m"),
    **kwargs: Any,
) -> AbstractPos3D:
    """AbstractPos2D -> Cartesian2D -> Cartesian3D -> AbstractPos3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1, 2], "km")

    >>> x2 = cx.vconvert(cx.vecs.CylindricalPos, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CylindricalPos (rho[km], phi[rad], z[km])
        [ 2.236  1.107 14.   ]>

    >>> x3 = cx.vconvert(cx.SphericalPos, x, z=u.Quantity(14, "km"))
    >>> x3
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                  theta=Angle(value=f32[], unit=Unit("rad")),
                  phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    >>> x3 = cx.vconvert(cx.vecs.MathSphericalPos, x, z=u.Quantity(14, "km"))
    >>> x3
    MathSphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                      theta=Angle(value=f32[], unit=Unit("rad")),
                      phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    """
    cart2 = vconvert(CartesianPos2D, current)
    cart3 = vconvert(CartesianPos3D, cart2, z=z)
    return vconvert(target, cart3)


@dispatch.multi(
    (type[CartesianPos3D], PolarPos),
)
def vconvert(
    target: type[AbstractPos3D],
    current: AbstractPos2D,
    /,
    z: AbstractQuantity = Quantity(0, "m"),
    **kwargs: Any,
) -> AbstractPos3D:
    """AbstractPos2D -> PolarPos -> Cylindrical -> AbstractPos3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> x2 = cx.vconvert(cx.CartesianPos3D, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CartesianPos3D (x[km], y[km], z[km])
        [ 0.985  0.174 14.   ]>

    """
    polar = vconvert(PolarPos, current)
    cyl = vconvert(CylindricalPos, polar, z=z)
    return vconvert(target, cyl)


# =============================================================================
# CartesianPos2D


# -----------------------------------------------
# 1D


@dispatch
def vconvert(
    target: type[CartesianPos1D], current: CartesianPos2D, /, **kwargs: Any
) -> CartesianPos1D:
    """CartesianPos2D -> CartesianPos1D.

    The `y` coordinate is dropped.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1, 2], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos1D, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CartesianPos1D (x[km])
        [1]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def vconvert(
    target: type[RadialPos], current: CartesianPos2D, /, **kwargs: Any
) -> RadialPos:
    """CartesianPos2D -> RadialPos.

    The `x` and `y` coordinates are converted to the radial coordinate `r`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1, 2], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.RadialPos, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <RadialPos (r[km])
        [2.236]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=jnp.sqrt(current.x**2 + current.y**2))


# -----------------------------------------------
# 3D


@dispatch
def vconvert(
    target: type[CartesianPos3D],
    current: CartesianPos2D,
    /,
    *,
    z: AbstractQuantity = Quantity(0, "m"),
    **kwargs: Any,
) -> CartesianPos3D:
    """CartesianPos2D -> CartesianPos3D.

    The `x` and `y` coordinates are converted to the `x` and `y` coordinates of
    the 3D system.  The `z` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1, 2], "km")

    >>> x2 = cx.vconvert(cx.CartesianPos3D, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CartesianPos3D (x[km], y[km], z[km])
        [ 1  2 14]>

    """
    return target(x=current.x, y=current.y, z=z)


# =============================================================================
# PolarPos

# -----------------------------------------------
# 1D


@dispatch
def vconvert(
    target: type[CartesianPos1D], current: PolarPos, /, **kwargs: Any
) -> CartesianPos1D:
    """PolarPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(x2)
    <CartesianPos1D (x[km])
        [0.985]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * jnp.cos(current.phi))


@dispatch
def vconvert(target: type[RadialPos], current: PolarPos, /, **kwargs: Any) -> RadialPos:
    """PolarPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.RadialPos, x)
    >>> print(x2)
    <RadialPos (r[km])
        [1]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 3D


@dispatch
def vconvert(
    target: type[SphericalPos],
    current: PolarPos,
    /,
    theta: Quantity["angle"] = Quantity(0, "radian"),
    **kwargs: Any,
) -> SphericalPos:
    """PolarPos -> SphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> x2 = cx.vconvert(cx.SphericalPos, x, theta=u.Quantity(14, "deg"))
    >>> print(x2)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [ 1 14 10]>

    """
    return target.from_(r=current.r, theta=theta, phi=current.phi)


@dispatch
def vconvert(
    target: type[MathSphericalPos],
    current: PolarPos,
    /,
    phi: Quantity["angle"] = Quantity(0, "radian"),
    **kwargs: Any,
) -> MathSphericalPos:
    """PolarPos -> MathSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> x2 = cx.vconvert(cx.vecs.MathSphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x2)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [ 1 10 14]>

    """
    return target.from_(r=current.r, phi=phi, theta=current.phi)


@dispatch
def vconvert(
    target: type[CylindricalPos],
    current: PolarPos,
    /,
    *,
    z: Quantity["length"] = Quantity(0, "m"),
    **kwargs: Any,
) -> CylindricalPos:
    """PolarPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> x2 = cx.vconvert(cx.vecs.CylindricalPos, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CylindricalPos (rho[km], phi[deg], z[km])
        [ 1 10 14]>

    """
    return target(rho=current.r, phi=current.phi, z=z)
