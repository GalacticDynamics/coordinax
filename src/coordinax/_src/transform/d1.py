"""Transformations from 1D."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.numpy as jnp
from unxt import Quantity

from coordinax._src.d1.cartesian import CartesianPos1D
from coordinax._src.d1.radial import RadialPos
from coordinax._src.d2.cartesian import CartesianPos2D
from coordinax._src.d2.polar import PolarPos
from coordinax._src.d3.cartesian import CartesianPos3D
from coordinax._src.d3.cylindrical import CylindricalPos
from coordinax._src.d3.mathspherical import MathSphericalPos
from coordinax._src.d3.spherical import SphericalPos

# =============================================================================
# CartesianPos1D


# -----------------------------------------------
# to 2D


@dispatch
def represent_as(
    current: CartesianPos1D,
    target: type[CartesianPos2D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPos2D:
    """CartesianPos1D -> CartesianPos2D.

    The `x` coordinate is converted to the `x` coordinate of the 2D system.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPos1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPos2D)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPos3D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.x, y=y)


@dispatch
def represent_as(
    current: CartesianPos1D,
    target: type[PolarPos],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> PolarPos:
    """CartesianPos1D -> PolarPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPos1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.PolarPos, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.x, phi=phi)


# -----------------------------------------------
# to 3D


@dispatch
def represent_as(
    current: CartesianPos1D,
    target: type[CartesianPos3D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPos3D:
    """CartesianPos1D -> CartesianPos3D.

    The `x` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPos1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPos3D)
    >>> x2
    CartesianPos3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPos3D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.x, y=y, z=z)


@dispatch
def represent_as(
    current: CartesianPos1D,
    target: type[SphericalPos] | type[MathSphericalPos],
    /,
    *,
    theta: Quantity = Quantity(0.0, "radian"),
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> SphericalPos | MathSphericalPos:
    """CartesianPos1D -> SphericalPos | MathSphericalPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    SphericalPos:

    >>> x = cx.CartesianPos1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.SphericalPos)
    >>> x2
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                       theta=Quantity[...](value=f32[], unit=Unit("deg")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.SphericalPos, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    MathSphericalPos:
    Note that ``theta`` and ``phi`` have different meanings in this context.

    >>> x2 = cx.represent_as(x, cx.MathSphericalPos)
    >>> x2
    MathSphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("rad")),
                         phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.MathSphericalPos, phi=Quantity(14, "deg"))
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    x, theta, phi = jnp.broadcast_arrays(current.x, theta, phi)
    return target.from_(r=x, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: CartesianPos1D,
    target: type[CylindricalPos],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CylindricalPos:
    """CartesianPos1D -> CylindricalPos.

    The `x` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPos1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CylindricalPos)
    >>> x2
    CylindricalPos( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CylindricalPos, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(rho=current.x, phi=phi, z=z)


# =============================================================================
# RadialPos

# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: RadialPos,
    target: type[CartesianPos2D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPos2D:
    """RadialPos -> CartesianPos2D.

    The `r` coordinate is converted to the cartesian coordinate `x`.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPos(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPos2D)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPos2D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.r, y=y)


@dispatch
def represent_as(
    current: RadialPos,
    target: type[PolarPos],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> PolarPos:
    """RadialPos -> PolarPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPos(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.PolarPos, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.r, phi=phi)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: RadialPos,
    target: type[CartesianPos3D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPos3D:
    """RadialPos -> CartesianPos3D.

    The `r` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPos(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPos3D)
    >>> x2
    CartesianPos3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPos3D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.r, y=y, z=z)


@dispatch
def represent_as(
    current: RadialPos,
    target: type[SphericalPos] | type[MathSphericalPos],
    /,
    *,
    theta: Quantity = Quantity(0.0, "radian"),
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> SphericalPos | MathSphericalPos:
    """RadialPos -> SphericalPos | MathSphericalPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPos(r=Quantity(1.0, "km"))

    SphericalPos:

    >>> x2 = cx.represent_as(x, cx.SphericalPos)
    >>> x2
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                       theta=Quantity[...](value=f32[], unit=Unit("deg")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.SphericalPos, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    MathSphericalPos:

    >>> x2 = cx.represent_as(x, cx.MathSphericalPos)
    >>> x2
    MathSphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                           theta=Quantity[...](value=f32[], unit=Unit("rad")),
                           phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.MathSphericalPos, phi=Quantity(14, "deg"))
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    r, theta, phi = jnp.broadcast_arrays(current.r, theta, phi)
    return target.from_(r=r, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: RadialPos,
    target: type[CylindricalPos],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CylindricalPos:
    """RadialPos -> CylindricalPos.

    The `r` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPos(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CylindricalPos)
    >>> x2
    CylindricalPos( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CylindricalPos, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(rho=current.r, phi=phi, z=z)
