"""Transformations from 1D."""

__all__ = ["represent_as"]

from typing import Any

from plum import dispatch

import quaxed.array_api as xp
from unxt import Quantity

from coordinax._coordinax.d1.cartesian import CartesianPosition1D
from coordinax._coordinax.d1.radial import RadialPosition
from coordinax._coordinax.d2.cartesian import CartesianPosition2D
from coordinax._coordinax.d2.polar import PolarPosition
from coordinax._coordinax.d3.cartesian import CartesianPosition3D
from coordinax._coordinax.d3.cylindrical import CylindricalPosition
from coordinax._coordinax.d3.spherical import MathSphericalPosition, SphericalPosition

# =============================================================================
# CartesianPosition1D


# -----------------------------------------------
# to 2D


@dispatch
def represent_as(
    current: CartesianPosition1D,
    target: type[CartesianPosition2D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPosition2D:
    """CartesianPosition1D -> CartesianPosition2D.

    The `x` coordinate is converted to the `x` coordinate of the 2D system.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPosition2D)
    >>> x2
    CartesianPosition2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPosition3D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.x, y=y)


@dispatch
def represent_as(
    current: CartesianPosition1D,
    target: type[PolarPosition],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> PolarPosition:
    """CartesianPosition1D -> PolarPosition.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.PolarPosition)
    >>> x2
    PolarPosition( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.PolarPosition, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.x, phi=phi)


# -----------------------------------------------
# to 3D


@dispatch
def represent_as(
    current: CartesianPosition1D,
    target: type[CartesianPosition3D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPosition3D:
    """CartesianPosition1D -> CartesianPosition3D.

    The `x` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPosition3D)
    >>> x2
    CartesianPosition3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPosition3D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.x, y=y, z=z)


@dispatch
def represent_as(
    current: CartesianPosition1D,
    target: type[SphericalPosition] | type[MathSphericalPosition],
    /,
    *,
    theta: Quantity = Quantity(0.0, "radian"),
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> SphericalPosition | MathSphericalPosition:
    """CartesianPosition1D -> SphericalPosition | MathSphericalPosition.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    SphericalPosition:

    >>> x = cx.CartesianPosition1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.SphericalPosition)
    >>> x2
    SphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                       theta=Quantity[...](value=f32[], unit=Unit("deg")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.SphericalPosition, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    MathSphericalPosition:
    Note that ``theta`` and ``phi`` have different meanings in this context.

    >>> x2 = cx.represent_as(x, cx.MathSphericalPosition)
    >>> x2
    MathSphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("rad")),
                         phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.MathSphericalPosition, phi=Quantity(14, "deg"))
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    x, theta, phi = xp.broadcast_arrays(current.x, theta, phi)
    return target.constructor(r=x, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: CartesianPosition1D,
    target: type[CylindricalPosition],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CylindricalPosition:
    """CartesianPosition1D -> CylindricalPosition.

    The `x` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition1D(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CylindricalPosition)
    >>> x2
    CylindricalPosition( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CylindricalPosition, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(rho=current.x, phi=phi, z=z)


# =============================================================================
# RadialPosition

# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: RadialPosition,
    target: type[CartesianPosition2D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPosition2D:
    """RadialPosition -> CartesianPosition2D.

    The `r` coordinate is converted to the cartesian coordinate `x`.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPosition(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPosition2D)
    >>> x2
    CartesianPosition2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPosition2D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.r, y=y)


@dispatch
def represent_as(
    current: RadialPosition,
    target: type[PolarPosition],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> PolarPosition:
    """RadialPosition -> PolarPosition.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPosition(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.PolarPosition)
    >>> x2
    PolarPosition( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.PolarPosition, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.r, phi=phi)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: RadialPosition,
    target: type[CartesianPosition3D],
    /,
    *,
    y: Quantity = Quantity(0.0, "m"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPosition3D:
    """RadialPosition -> CartesianPosition3D.

    The `r` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPosition(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CartesianPosition3D)
    >>> x2
    CartesianPosition3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CartesianPosition3D, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.r, y=y, z=z)


@dispatch
def represent_as(
    current: RadialPosition,
    target: type[SphericalPosition] | type[MathSphericalPosition],
    /,
    *,
    theta: Quantity = Quantity(0.0, "radian"),
    phi: Quantity = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> SphericalPosition | MathSphericalPosition:
    """RadialPosition -> SphericalPosition | MathSphericalPosition.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPosition(r=Quantity(1.0, "km"))

    SphericalPosition:

    >>> x2 = cx.represent_as(x, cx.SphericalPosition)
    >>> x2
    SphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                       theta=Quantity[...](value=f32[], unit=Unit("deg")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.SphericalPosition, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    MathSphericalPosition:

    >>> x2 = cx.represent_as(x, cx.MathSphericalPosition)
    >>> x2
    MathSphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                           theta=Quantity[...](value=f32[], unit=Unit("rad")),
                           phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='deg')

    >>> x3 = cx.represent_as(x, cx.MathSphericalPosition, phi=Quantity(14, "deg"))
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    r, theta, phi = xp.broadcast_arrays(current.r, theta, phi)
    return target.constructor(r=r, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: RadialPosition,
    target: type[CylindricalPosition],
    /,
    *,
    phi: Quantity = Quantity(0.0, "radian"),
    z: Quantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CylindricalPosition:
    """RadialPosition -> CylindricalPosition.

    The `r` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialPosition(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CylindricalPosition)
    >>> x2
    CylindricalPosition( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CylindricalPosition, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(rho=current.r, phi=phi, z=z)
