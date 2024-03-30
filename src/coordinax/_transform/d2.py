"""Transformations between representations."""

__all__ = ["represent_as"]

from typing import Any
from warnings import warn

import astropy.units as u
from plum import dispatch

import quaxed.array_api as xp
from unxt import Quantity

from coordinax._d1.builtin import Cartesian1DVector, RadialVector
from coordinax._d2.base import Abstract2DVector
from coordinax._d2.builtin import Cartesian2DVector, PolarVector
from coordinax._d3.base import Abstract3DVector
from coordinax._d3.builtin import Cartesian3DVector, CylindricalVector
from coordinax._d3.sphere import MathSphericalVector, SphericalVector
from coordinax._exceptions import IrreversibleDimensionChange


@dispatch.multi(
    (Cartesian2DVector, type[CylindricalVector]),
    (Cartesian2DVector, type[SphericalVector]),
    (Cartesian2DVector, type[MathSphericalVector]),
)
def represent_as(
    current: Abstract2DVector,
    target: type[Abstract3DVector],
    /,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Abstract3DVector:
    """Abstract2DVector -> Cartesian2D -> Cartesian3D -> Abstract3DVector.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian2DVector.constructor(Quantity([1.0, 2.0], "km"))

    >>> x2 = cx.represent_as(x, cx.CylindricalVector, z=Quantity(14, "km"))
    >>> x2
    CylindricalVector( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    >>> x3 = cx.represent_as(x, cx.SphericalVector, z=Quantity(14, "km"))
    >>> x3
    SphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                     theta=Quantity[...](value=f32[], unit=Unit("rad")),
                     phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    >>> x3 = cx.represent_as(x, cx.MathSphericalVector, z=Quantity(14, "km"))
    >>> x3
    MathSphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("rad")),
                         phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    """
    cart2 = represent_as(current, Cartesian2DVector)
    cart3 = represent_as(cart2, Cartesian3DVector, z=z)
    return represent_as(cart3, target)


@dispatch.multi(
    (PolarVector, type[Cartesian3DVector]),
)
def represent_as(
    current: Abstract2DVector,
    target: type[Abstract3DVector],
    /,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Abstract3DVector:
    """Abstract2DVector -> PolarVector -> Cylindrical -> Abstract3DVector.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarVector(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.Cartesian3DVector, z=Quantity(14, "km"))
    >>> x2
    Cartesian3DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    polar = represent_as(current, PolarVector)
    cyl = represent_as(polar, CylindricalVector, z=z)
    return represent_as(cyl, target)


# =============================================================================
# Cartesian2DVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: Cartesian2DVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """Cartesian2DVector -> Cartesian1DVector.

    The `y` coordinate is dropped.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian2DVector.constructor(Quantity([1.0, 2.0], "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian1DVector, z=Quantity(14, "km"))
    >>> x2
    Cartesian1DVector( x=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.x
    Quantity['length'](Array(1., dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def represent_as(
    current: Cartesian2DVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """Cartesian2DVector -> RadialVector.

    The `x` and `y` coordinates are converted to the radial coordinate `r`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian2DVector.constructor(Quantity([1.0, 2.0], "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialVector, z=Quantity(14, "km"))
    >>> x2
    RadialVector(r=Distance(value=f32[], unit=Unit("km")))
    >>> x2.r
    Distance(Array(2.236068, dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=xp.sqrt(current.x**2 + current.y**2))


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: Cartesian2DVector,
    target: type[Cartesian3DVector],
    /,
    *,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian3DVector:
    """Cartesian2DVector -> Cartesian3DVector.

    The `x` and `y` coordinates are converted to the `x` and `y` coordinates of
    the 3D system.  The `z` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian2DVector.constructor(Quantity([1.0, 2.0], "km"))

    >>> x2 = cx.represent_as(x, cx.Cartesian3DVector, z=Quantity(14, "km"))
    >>> x2
    Cartesian3DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.x, y=current.y, z=z)


# =============================================================================
# PolarVector

# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: PolarVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """PolarVector -> Cartesian1DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarVector(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian1DVector)
    >>> x2
    Cartesian1DVector( x=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.x
    Quantity['length'](Array(0.9848077, dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * xp.cos(current.phi))


@dispatch
def represent_as(
    current: PolarVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """PolarVector -> RadialVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarVector(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialVector)
    >>> x2
    RadialVector(r=Distance(value=f32[], unit=Unit("km")))
    >>> x2.r
    Distance(Array(1., dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: PolarVector,
    target: type[SphericalVector],
    /,
    theta: Quantity["angle"] = Quantity(0.0, u.radian),  # type: ignore[name-defined]
    **kwargs: Any,
) -> SphericalVector:
    """PolarVector -> SphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarVector(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.SphericalVector, theta=Quantity(14, "deg"))
    >>> x2
    SphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                     theta=Quantity[...](value=f32[], unit=Unit("deg")),
                     phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.r, theta=theta, phi=current.phi)


@dispatch
def represent_as(
    current: PolarVector,
    target: type[MathSphericalVector],
    /,
    phi: Quantity["angle"] = Quantity(0.0, u.radian),  # type: ignore[name-defined]
    **kwargs: Any,
) -> MathSphericalVector:
    """PolarVector -> MathSphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarVector(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.MathSphericalVector, phi=Quantity(14, "deg"))
    >>> x2
    MathSphericalVector( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("deg")),
                         phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target(r=current.r, phi=phi, theta=current.phi)


@dispatch
def represent_as(
    current: PolarVector,
    target: type[CylindricalVector],
    /,
    *,
    z: Quantity["length"] = Quantity(0.0, u.m),  # type: ignore[name-defined]
    **kwargs: Any,
) -> CylindricalVector:
    """PolarVector -> CylindricalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarVector(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.CylindricalVector, z=Quantity(14, "km"))
    >>> x2
    CylindricalVector( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("deg")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(rho=current.r, phi=current.phi, z=z)
