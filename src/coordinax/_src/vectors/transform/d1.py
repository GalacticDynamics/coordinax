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
    y: u.Quantity = u.Quantity(0.0, "m"),
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
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.vconvert(cx.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.x, y=y)


@dispatch
def vconvert(
    target: type[PolarPos],
    current: CartesianPos1D,
    /,
    *,
    phi: u.Quantity = u.Quantity(0.0, "radian"),
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
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.vconvert(cx.vecs.PolarPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')

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
    y: u.Quantity = u.Quantity(0.0, "m"),
    z: u.Quantity = u.Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPos3D:
    """CartesianPos1D -> CartesianPos3D.

    The `x` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.CartesianPos3D, x)
    >>> x2
    CartesianPos3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("m")),
                    z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.vconvert(cx.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.x, y=y, z=z)


@dispatch
def vconvert(
    target: type[SphericalPos] | type[MathSphericalPos],
    current: CartesianPos1D,
    /,
    *,
    theta: u.Quantity = u.Quantity(0.0, "radian"),
    phi: u.Quantity = u.Quantity(0.0, "radian"),
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

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.SphericalPos, x)
    >>> x2
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                  theta=Angle(value=f32[], unit=Unit("deg")),
                  phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Angle(Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.vconvert(cx.SphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')
    >>> x2.theta
    Angle(Array(0., dtype=float32), unit='deg')

    MathSphericalPos:
    Note that ``theta`` and ``phi`` have different meanings in this context.

    >>> x2 = cx.vconvert(cx.vecs.MathSphericalPos, x)
    >>> x2
    MathSphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                      theta=Angle(value=f32[], unit=Unit("rad")),
                      phi=Angle(value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.vconvert(cx.vecs.MathSphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.theta
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')

    """
    x, theta, phi = jnp.broadcast_arrays(current.x, theta, phi)
    return target.from_(r=x, theta=theta, phi=phi)


@dispatch
def vconvert(
    target: type[CylindricalPos],
    current: CartesianPos1D,
    /,
    *,
    phi: u.Quantity = u.Quantity(0.0, "radian"),
    z: u.Quantity = u.Quantity(0.0, "m"),
    **kwargs: Any,
) -> CylindricalPos:
    """CartesianPos1D -> CylindricalPos.

    The `x` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.vecs.CylindricalPos, x)
    >>> x2
    CylindricalPos( rho=Quantity[...](value=f32[], unit=Unit("km")),
                    phi=Angle(value=f32[], unit=Unit("rad")),
                    z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.vconvert(cx.vecs.CylindricalPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

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
    y: u.Quantity = u.Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPos2D:
    """RadialPos -> CartesianPos2D.

    The `r` coordinate is converted to the cartesian coordinate `x`.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.vconvert(cx.vecs.CartesianPos2D, x, y=u.Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.r, y=y)


@dispatch
def vconvert(
    target: type[PolarPos],
    current: RadialPos,
    /,
    *,
    phi: u.Quantity = u.Quantity(0.0, "radian"),
    **kwargs: Any,
) -> PolarPos:
    """RadialPos -> PolarPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1.0, "km"))
    >>> x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.vconvert(cx.vecs.PolarPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')

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
    y: u.Quantity = u.Quantity(0.0, "m"),
    z: u.Quantity = u.Quantity(0.0, "m"),
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
    >>> x2
    CartesianPos3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("m")),
                    z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.vconvert(cx.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.r, y=y, z=z)


@dispatch
def vconvert(
    target: type[SphericalPos] | type[MathSphericalPos],
    current: RadialPos,
    /,
    *,
    theta: u.Quantity = u.Quantity(0.0, "radian"),
    phi: u.Quantity = u.Quantity(0.0, "radian"),
    **kwargs: Any,
) -> SphericalPos | MathSphericalPos:
    """RadialPos -> SphericalPos | MathSphericalPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.RadialPos(r=u.Quantity(1.0, "km"))

    SphericalPos:

    >>> x2 = cx.vconvert(cx.SphericalPos, x)
    >>> x2
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                  theta=Angle(value=f32[], unit=Unit("deg")),
                  phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Angle(Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.vconvert(cx.SphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')
    >>> x3.theta
    Angle(Array(0., dtype=float32), unit='deg')

    MathSphericalPos:

    >>> x2 = cx.vconvert(cx.vecs.MathSphericalPos, x)
    >>> x2
    MathSphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                      theta=Angle(value=f32[], unit=Unit("rad")),
                      phi=Angle(value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.vconvert(cx.vecs.MathSphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.theta
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')

    """
    r, theta, phi = jnp.broadcast_arrays(current.r, theta, phi)
    return target.from_(r=r, theta=theta, phi=phi)


@dispatch
def vconvert(
    target: type[CylindricalPos],
    current: RadialPos,
    /,
    *,
    phi: u.Quantity = u.Quantity(0.0, "radian"),
    z: u.Quantity = u.Quantity(0.0, "m"),
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
    >>> x2
    CylindricalPos( rho=Quantity[...](value=f32[], unit=Unit("km")),
                    phi=Angle(value=f32[], unit=Unit("rad")),
                    z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Angle(Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.vconvert(cx.vecs.CylindricalPos, x, phi=u.Quantity(14, "deg"))
    >>> x3.phi
    Angle(Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(rho=current.r, phi=phi, z=z)
