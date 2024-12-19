"""Transformations from 1D."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.numpy as jnp
import unxt as u

from coordinax._src.vectors.d1 import CartesianPos1D, RadialPos
from coordinax._src.vectors.d2 import CartesianPos2D, PolarPos
from coordinax._src.vectors.d3 import (
    CartesianPos3D,
    CylindricalPos,
    MathSphericalPos,
    SphericalPos,
)

# =============================================================================
# CartesianPos1D


# -----------------------------------------------
# to 2D


@dispatch
def vconvert(
    target: type[CartesianPos2D],
    current: CartesianPos1D,
    /,
    *,
    y: u.Quantity = u.Quantity(0, "m"),
    **kwargs: Any,
) -> CartesianPos2D:
    """CartesianPos1D -> CartesianPos2D.

    The `x` coordinate is converted to the `x` coordinate of the 2D system.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D (x[km], y[m])
        [1. 0.]>

    >>> x3 = cx.vconvert(cx.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> print(x3)
    <CartesianPos3D (x[km], y[km], z[m])
        [ 1. 14.  0.]>

    """
    return target(x=current.x, y=y)


@dispatch
def vconvert(
    target: type[PolarPos],
    current: CartesianPos1D,
    /,
    *,
    phi: u.Quantity = u.Quantity(0, "radian"),
    **kwargs: Any,
) -> PolarPos:
    """CartesianPos1D -> PolarPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(x2)
    <PolarPos (r[km], phi[rad])
        [1. 0.]>

    >>> x3 = cx.vconvert(cx.vecs.PolarPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <PolarPos (r[km], phi[deg])
        [ 1. 14.]>

    """
    return target(r=current.x, phi=phi)


# -----------------------------------------------
# to 3D


@dispatch
def vconvert(
    target: type[CartesianPos3D],
    current: CartesianPos1D,
    /,
    *,
    y: u.Quantity = u.Quantity(0, "m"),
    z: u.Quantity = u.Quantity(0, "m"),
    **kwargs: Any,
) -> CartesianPos3D:
    """CartesianPos1D -> CartesianPos3D.

    The `x` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1, "km"))
    >>> x2 = cx.vconvert(cx.CartesianPos3D, x)
    >>> print(x2)
    <CartesianPos3D (x[km], y[m], z[m])
        [1 0 0]>

    >>> x3 = cx.vconvert(cx.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> print(x3)
    <CartesianPos3D (x[km], y[km], z[m])
        [ 1 14  0]>

    """
    return target(x=current.x, y=y, z=z)


@dispatch
def vconvert(
    target: type[SphericalPos] | type[MathSphericalPos],
    current: CartesianPos1D,
    /,
    *,
    theta: u.Quantity = u.Quantity(0, "radian"),
    phi: u.Quantity = u.Quantity(0, "radian"),
    **kwargs: Any,
) -> SphericalPos | MathSphericalPos:
    """CartesianPos1D -> SphericalPos | MathSphericalPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    SphericalPos:

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1, "km"))
    >>> x2 = cx.vconvert(cx.SphericalPos, x)
    >>> print(x2)
    <SphericalPos (r[km], theta[deg], phi[rad])
        [1. 0. 0.]>

    >>> x3 = cx.vconvert(cx.SphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [ 1.  0. 14.]>

    MathSphericalPos:
    Note that ``theta`` and ``phi`` have different meanings in this context.

    >>> x2 = cx.vconvert(cx.vecs.MathSphericalPos, x)
    >>> print(x2)
    <MathSphericalPos (r[km], theta[rad], phi[deg])
        [1. 0. 0.]>

    >>> x3 = cx.vconvert(cx.vecs.MathSphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <MathSphericalPos (r[km], theta[rad], phi[deg])
        [ 1.  0. 14.]>

    """
    x, theta, phi = jnp.broadcast_arrays(current.x, theta, phi)
    return target.from_(r=x, theta=theta, phi=phi)


@dispatch
def vconvert(
    target: type[CylindricalPos],
    current: CartesianPos1D,
    /,
    *,
    phi: u.Quantity = u.Quantity(0, "radian"),
    z: u.Quantity = u.Quantity(0, "m"),
    **kwargs: Any,
) -> CylindricalPos:
    """CartesianPos1D -> CylindricalPos.

    The `x` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1, "km"))
    >>> x2 = cx.vconvert(cx.vecs.CylindricalPos, x)
    >>> print(x2)
    <CylindricalPos (rho[km], phi[rad], z[m])
        [1. 0. 0.]>

    >>> x3 = cx.vconvert(cx.vecs.CylindricalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <CylindricalPos (rho[km], phi[deg], z[m])
        [ 1 14  0]>

    """
    return target(rho=current.x, phi=phi, z=z)


# =============================================================================
# RadialPos

# -----------------------------------------------
# 2D


@dispatch
def vconvert(
    target: type[CartesianPos2D],
    current: RadialPos,
    /,
    *,
    y: u.Quantity = u.Quantity(0, "m"),
    **kwargs: Any,
) -> CartesianPos2D:
    """RadialPos -> CartesianPos2D.

    The `r` coordinate is converted to the cartesian coordinate `x`.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1, "km"))
    >>> x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D (x[km], y[m])
        [1 0]>

    >>> x3 = cx.vconvert(cx.vecs.CartesianPos2D, x, y=u.Quantity(14, "km"))
    >>> print(x3)
    <CartesianPos2D (x[km], y[km])
        [ 1 14]>

    """
    return target(x=current.r, y=y)


@dispatch
def vconvert(
    target: type[PolarPos],
    current: RadialPos,
    /,
    *,
    phi: u.Quantity = u.Quantity(0, "radian"),
    **kwargs: Any,
) -> PolarPos:
    """RadialPos -> PolarPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1, "km"))
    >>> x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(x2)
    <PolarPos (r[km], phi[rad])
        [1. 0.]>

    >>> x3 = cx.vconvert(cx.vecs.PolarPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <PolarPos (r[km], phi[deg])
        [ 1 14]>

    """
    return target(r=current.r, phi=phi)


# -----------------------------------------------
# 3D


@dispatch
def vconvert(
    target: type[CartesianPos3D],
    current: RadialPos,
    /,
    *,
    y: u.Quantity = u.Quantity(0, "m"),
    z: u.Quantity = u.Quantity(0, "m"),
    **kwargs: Any,
) -> CartesianPos3D:
    """RadialPos -> CartesianPos3D.

    The `r` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.CartesianPos3D, x)
    >>> print(x2)
    <CartesianPos3D (x[km], y[m], z[m])
        [1. 0. 0.]>

    >>> x3 = cx.vconvert(cx.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> print(x3)
    <CartesianPos3D (x[km], y[km], z[m])
        [ 1. 14.  0.]>

    """
    return target(x=current.r, y=y, z=z)


@dispatch
def vconvert(
    target: type[SphericalPos] | type[MathSphericalPos],
    current: RadialPos,
    /,
    *,
    theta: u.Quantity = u.Quantity(0, "radian"),
    phi: u.Quantity = u.Quantity(0, "radian"),
    **kwargs: Any,
) -> SphericalPos | MathSphericalPos:
    """RadialPos -> SphericalPos | MathSphericalPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1, "km"))

    SphericalPos:

    >>> x2 = cx.vconvert(cx.SphericalPos, x)
    >>> print(x2)
    <SphericalPos (r[km], theta[deg], phi[rad])
        [1. 0. 0.]>

    >>> x3 = cx.vconvert(cx.SphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [ 1.  0. 14.]>

    MathSphericalPos:

    >>> x2 = cx.vconvert(cx.vecs.MathSphericalPos, x)
    >>> print(x2)
    <MathSphericalPos (r[km], theta[rad], phi[deg])
        [1. 0. 0.]>

    >>> x3 = cx.vconvert(cx.vecs.MathSphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <MathSphericalPos (r[km], theta[rad], phi[deg])
        [ 1.  0. 14.]>

    """
    r, theta, phi = jnp.broadcast_arrays(current.r, theta, phi)
    return target.from_(r=r, theta=theta, phi=phi)


@dispatch
def vconvert(
    target: type[CylindricalPos],
    current: RadialPos,
    /,
    *,
    phi: u.Quantity = u.Quantity(0, "radian"),
    z: u.Quantity = u.Quantity(0, "m"),
    **kwargs: Any,
) -> CylindricalPos:
    """RadialPos -> CylindricalPos.

    The `r` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.vecs.CylindricalPos, x)
    >>> print(x2)
    <CylindricalPos (rho[km], phi[rad], z[m])
        [1. 0. 0.]>

    >>> x3 = cx.vconvert(cx.vecs.CylindricalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <CylindricalPos (rho[km], phi[deg], z[m])
        [ 1. 14.  0.]>

    """
    return target(rho=current.r, phi=phi, z=z)
