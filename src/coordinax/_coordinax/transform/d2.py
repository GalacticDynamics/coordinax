"""Transformations between representations."""

__all__ = ["represent_as"]

from typing import Any
from warnings import warn

import astropy.units as u
from plum import dispatch

import quaxed.array_api as xp
from unxt import Quantity

from coordinax._coordinax.d1.cartesian import CartesianPosition1D
from coordinax._coordinax.d1.radial import RadialPosition
from coordinax._coordinax.d2.base import AbstractPosition2D
from coordinax._coordinax.d2.cartesian import CartesianPosition2D
from coordinax._coordinax.d2.polar import PolarPosition
from coordinax._coordinax.d3.base import AbstractPosition3D
from coordinax._coordinax.d3.cartesian import CartesianPosition3D
from coordinax._coordinax.d3.cylindrical import CylindricalPosition
from coordinax._coordinax.d3.spherical import MathSphericalPosition, SphericalPosition
from coordinax._coordinax.exceptions import IrreversibleDimensionChange


@dispatch.multi(
    (CartesianPosition2D, type[CylindricalPosition]),
    (CartesianPosition2D, type[SphericalPosition]),
    (CartesianPosition2D, type[MathSphericalPosition]),
)
def represent_as(
    current: AbstractPosition2D,
    target: type[AbstractPosition3D],
    /,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> AbstractPosition3D:
    """AbstractPosition2D -> Cartesian2D -> Cartesian3D -> AbstractPosition3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")

    >>> x2 = cx.represent_as(x, cx.CylindricalPosition, z=Quantity(14, "km"))
    >>> x2
    CylindricalPosition( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("rad")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    >>> x3 = cx.represent_as(x, cx.SphericalPosition, z=Quantity(14, "km"))
    >>> x3
    SphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                     theta=Quantity[...](value=f32[], unit=Unit("rad")),
                     phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    >>> x3 = cx.represent_as(x, cx.MathSphericalPosition, z=Quantity(14, "km"))
    >>> x3
    MathSphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("rad")),
                         phi=Quantity[...](value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    """
    cart2 = represent_as(current, CartesianPosition2D)
    cart3 = represent_as(cart2, CartesianPosition3D, z=z)
    return represent_as(cart3, target)


@dispatch.multi(
    (PolarPosition, type[CartesianPosition3D]),
)
def represent_as(
    current: AbstractPosition2D,
    target: type[AbstractPosition3D],
    /,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> AbstractPosition3D:
    """AbstractPosition2D -> PolarPosition -> Cylindrical -> AbstractPosition3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarPosition(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.CartesianPosition3D, z=Quantity(14, "km"))
    >>> x2
    CartesianPosition3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    polar = represent_as(current, PolarPosition)
    cyl = represent_as(polar, CylindricalPosition, z=z)
    return represent_as(cyl, target)


# =============================================================================
# CartesianPosition2D


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: CartesianPosition2D, target: type[CartesianPosition1D], /, **kwargs: Any
) -> CartesianPosition1D:
    """CartesianPosition2D -> CartesianPosition1D.

    The `y` coordinate is dropped.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPosition1D, z=Quantity(14, "km"))
    >>> x2
    CartesianPosition1D( x=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.x
    Quantity['length'](Array(1., dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def represent_as(
    current: CartesianPosition2D, target: type[RadialPosition], /, **kwargs: Any
) -> RadialPosition:
    """CartesianPosition2D -> RadialPosition.

    The `x` and `y` coordinates are converted to the radial coordinate `r`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialPosition, z=Quantity(14, "km"))
    >>> x2
    RadialPosition(r=Distance(value=f32[], unit=Unit("km")))
    >>> x2.r
    Distance(Array(2.236068, dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=xp.sqrt(current.x**2 + current.y**2))


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: CartesianPosition2D,
    target: type[CartesianPosition3D],
    /,
    *,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> CartesianPosition3D:
    """CartesianPosition2D -> CartesianPosition3D.

    The `x` and `y` coordinates are converted to the `x` and `y` coordinates of
    the 3D system.  The `z` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")

    >>> x2 = cx.represent_as(x, cx.CartesianPosition3D, z=Quantity(14, "km"))
    >>> x2
    CartesianPosition3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.x, y=current.y, z=z)


# =============================================================================
# PolarPosition

# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: PolarPosition, target: type[CartesianPosition1D], /, **kwargs: Any
) -> CartesianPosition1D:
    """PolarPosition -> CartesianPosition1D.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarPosition(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPosition1D)
    >>> x2
    CartesianPosition1D( x=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.x
    Quantity['length'](Array(0.9848077, dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * xp.cos(current.phi))


@dispatch
def represent_as(
    current: PolarPosition, target: type[RadialPosition], /, **kwargs: Any
) -> RadialPosition:
    """PolarPosition -> RadialPosition.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarPosition(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialPosition)
    >>> x2
    RadialPosition(r=Distance(value=f32[], unit=Unit("km")))
    >>> x2.r
    Distance(Array(1., dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: PolarPosition,
    target: type[SphericalPosition],
    /,
    theta: Quantity["angle"] = Quantity(0.0, u.radian),  # type: ignore[name-defined]
    **kwargs: Any,
) -> SphericalPosition:
    """PolarPosition -> SphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarPosition(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.SphericalPosition, theta=Quantity(14, "deg"))
    >>> x2
    SphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                     theta=Quantity[...](value=f32[], unit=Unit("deg")),
                     phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target.constructor(r=current.r, theta=theta, phi=current.phi)


@dispatch
def represent_as(
    current: PolarPosition,
    target: type[MathSphericalPosition],
    /,
    phi: Quantity["angle"] = Quantity(0.0, u.radian),  # type: ignore[name-defined]
    **kwargs: Any,
) -> MathSphericalPosition:
    """PolarPosition -> MathSphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarPosition(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.MathSphericalPosition, phi=Quantity(14, "deg"))
    >>> x2
    MathSphericalPosition( r=Distance(value=f32[], unit=Unit("km")),
                         theta=Quantity[...](value=f32[], unit=Unit("deg")),
                         phi=Quantity[...](value=f32[], unit=Unit("deg")) )
    >>> x2.phi
    Quantity['angle'](Array(14., dtype=float32), unit='deg')

    """
    return target.constructor(r=current.r, phi=phi, theta=current.phi)


@dispatch
def represent_as(
    current: PolarPosition,
    target: type[CylindricalPosition],
    /,
    *,
    z: Quantity["length"] = Quantity(0.0, u.m),  # type: ignore[name-defined]
    **kwargs: Any,
) -> CylindricalPosition:
    """PolarPosition -> CylindricalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.PolarPosition(r=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.CylindricalPosition, z=Quantity(14, "km"))
    >>> x2
    CylindricalPosition( rho=Quantity[...](value=f32[], unit=Unit("km")),
                       phi=Quantity[...](value=f32[], unit=Unit("deg")),
                       z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(rho=current.r, phi=current.phi, z=z)
