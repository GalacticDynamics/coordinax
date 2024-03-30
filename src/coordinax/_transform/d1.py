"""Transformations from 1D."""

__all__ = ["represent_as"]

from typing import Any

import astropy.units as u
from plum import dispatch

from unxt import Quantity

from coordinax._d1.builtin import Cartesian1DVector, RadialVector
from coordinax._d2.builtin import Cartesian2DVector, PolarVector
from coordinax._d3.builtin import Cartesian3DVector, CylindricalVector
from coordinax._d3.sphere import MathSphericalVector, SphericalVector

# =============================================================================
# Cartesian1DVector


# -----------------------------------------------
# to 2D


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[Cartesian2DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian2DVector:
    """Cartesian1DVector -> Cartesian2DVector.

    The `x` coordinate is converted to the `x` coordinate of the 2D system.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian1DVector(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.Cartesian2DVector)
    >>> x2
    Cartesian2DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.Cartesian3DVector, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.x, y=y)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[PolarVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> PolarVector:
    """Cartesian1DVector -> PolarVector.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian1DVector(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.PolarVector)
    >>> x2
    PolarVector( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.PolarVector, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.x, phi=phi)


# -----------------------------------------------
# to 3D


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[Cartesian3DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian3DVector:
    """Cartesian1DVector -> Cartesian3DVector.

    The `x` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian1DVector(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.Cartesian3DVector)
    >>> x2
    Cartesian3DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.Cartesian3DVector, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.x, y=y, z=z)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[SphericalVector] | type[MathSphericalVector],
    /,
    *,
    theta: Quantity = Quantity(0.0, u.radian),
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> SphericalVector | MathSphericalVector:
    """Cartesian1DVector -> SphericalVector | MathSphericalVector.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    SphericalVector:

    >>> x = cx.Cartesian1DVector(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.SphericalVector)
    >>> x2
    SphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                     theta=Quantity[...](value=f32[], unit=Unit("rad")),
                     phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.SphericalVector, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    MathSphericalVector:

    >>> x2 = cx.represent_as(x, cx.MathSphericalVector)
    >>> x2
    MathSphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("rad")),
                         phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.MathSphericalVector, phi=Quantity(14, "deg"))
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.x, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[CylindricalVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> CylindricalVector:
    """Cartesian1DVector -> CylindricalVector.

    The `x` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian1DVector(x=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CylindricalVector)
    >>> x2
    CylindricalVector( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CylindricalVector, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(rho=current.x, phi=phi, z=z)


# =============================================================================
# RadialVector

# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: RadialVector,
    target: type[Cartesian2DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian2DVector:
    """RadialVector -> Cartesian2DVector.

    The `r` coordinate is converted to the cartesian coordinate `x`.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialVector(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.Cartesian2DVector)
    >>> x2
    Cartesian2DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.Cartesian2DVector, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.r, y=y)


@dispatch
def represent_as(
    current: RadialVector,
    target: type[PolarVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> PolarVector:
    """RadialVector -> PolarVector.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialVector(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.PolarVector)
    >>> x2
    PolarVector( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.PolarVector, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.r, phi=phi)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: RadialVector,
    target: type[Cartesian3DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian3DVector:
    """RadialVector -> Cartesian3DVector.

    The `r` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialVector(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.Cartesian3DVector)
    >>> x2
    Cartesian3DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("m")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.y
    Quantity['length'](Array(0., dtype=float32), unit='m')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.Cartesian3DVector, y=Quantity(14, "km"))
    >>> x3.y
    Quantity['length'](Array(14., dtype=float32), unit='km')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(x=current.r, y=y, z=z)


@dispatch
def represent_as(
    current: RadialVector,
    target: type[SphericalVector] | type[MathSphericalVector],
    /,
    *,
    theta: Quantity = Quantity(0.0, u.radian),
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> SphericalVector | MathSphericalVector:
    """RadialVector -> SphericalVector | MathSphericalVector.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialVector(r=Quantity(1.0, "km"))

    SphericalVector:

    >>> x2 = cx.represent_as(x, cx.SphericalVector)
    >>> x2
    SphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                     theta=Quantity[...](value=f32[], unit=Unit("rad")),
                     phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.SphericalVector, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    MathSphericalVector:

    >>> x2 = cx.represent_as(x, cx.MathSphericalVector)
    >>> x2
    MathSphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("rad")),
                         phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x2.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')

    >>> x3 = cx.represent_as(x, cx.MathSphericalVector, phi=Quantity(14, "deg"))
    >>> x3.theta
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.r, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: RadialVector,
    target: type[CylindricalVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> CylindricalVector:
    """RadialVector -> CylindricalVector.

    The `r` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.RadialVector(r=Quantity(1.0, "km"))
    >>> x2 = cx.represent_as(x, cx.CylindricalVector)
    >>> x2
    CylindricalVector( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("m")) )
    >>> x2.phi
    Quantity['angle'](Array(0., dtype=float32), unit='rad')
    >>> x2.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> x3 = cx.represent_as(x, cx.CylindricalVector, phi=Quantity(14, "deg"))
    >>> x3.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')
    >>> x3.z
    Quantity['length'](Array(0., dtype=float32), unit='m')

    """
    return target(rho=current.r, phi=phi, z=z)
