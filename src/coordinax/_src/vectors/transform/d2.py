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
    (CartesianPos2D, type[CylindricalPos]),
    (CartesianPos2D, type[SphericalPos]),
    (CartesianPos2D, type[MathSphericalPos]),
)
def represent_as(
    current: AbstractPos2D,
    target: type[AbstractPos3D],
    /,
    z: AbstractQuantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> AbstractPos3D:
    """AbstractPos2D -> Cartesian2D -> Cartesian3D -> AbstractPos3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")

    >>> x2 = cx.represent_as(x, cx.vecs.CylindricalPos, z=u.Quantity(14, "km"))
    >>> x2
    CylindricalPos( rho=Quantity[...](value=f32[], unit=Unit("km")),
                    phi=Angle(value=f32[], unit=Unit("rad")),
                    z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    >>> x3 = cx.represent_as(x, cx.SphericalPos, z=u.Quantity(14, "km"))
    >>> x3
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                  theta=Angle(value=f32[], unit=Unit("rad")),
                  phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    >>> x3 = cx.represent_as(x, cx.vecs.MathSphericalPos, z=u.Quantity(14, "km"))
    >>> x3
    MathSphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                      theta=Angle(value=f32[], unit=Unit("rad")),
                      phi=Angle(value=f32[], unit=Unit("rad")) )
    >>> x3.r
    Distance(Array(14.177447, dtype=float32), unit='km')

    """
    cart2 = represent_as(current, CartesianPos2D)
    cart3 = represent_as(cart2, CartesianPos3D, z=z)
    return represent_as(cart3, target)


@dispatch.multi(
    (PolarPos, type[CartesianPos3D]),
)
def represent_as(
    current: AbstractPos2D,
    target: type[AbstractPos3D],
    /,
    z: AbstractQuantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> AbstractPos3D:
    """AbstractPos2D -> PolarPos -> Cylindrical -> AbstractPos3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.CartesianPos3D, z=u.Quantity(14, "km"))
    >>> x2
    CartesianPos3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("km")),
                    z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    polar = represent_as(current, PolarPos)
    cyl = represent_as(polar, CylindricalPos, z=z)
    return represent_as(cyl, target)


# =============================================================================
# CartesianPos2D


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: CartesianPos2D, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """CartesianPos2D -> CartesianPos1D.

    The `y` coordinate is dropped.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.vecs.CartesianPos1D, z=u.Quantity(14, "km"))
    >>> x2
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("km")))
    >>> x2.x
    Quantity['length'](Array(1., dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def represent_as(
    current: CartesianPos2D, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """CartesianPos2D -> RadialPos.

    The `x` and `y` coordinates are converted to the radial coordinate `r`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.vecs.RadialPos, z=u.Quantity(14, "km"))
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("km")))
    >>> x2.r
    Distance(Array(2.236068, dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=jnp.sqrt(current.x**2 + current.y**2))


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: CartesianPos2D,
    target: type[CartesianPos3D],
    /,
    *,
    z: AbstractQuantity = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CartesianPos3D:
    """CartesianPos2D -> CartesianPos3D.

    The `x` and `y` coordinates are converted to the `x` and `y` coordinates of
    the 3D system.  The `z` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")

    >>> x2 = cx.represent_as(x, cx.CartesianPos3D, z=u.Quantity(14, "km"))
    >>> x2
    CartesianPos3D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("km")),
                    z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(x=current.x, y=current.y, z=z)


# =============================================================================
# PolarPos

# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: PolarPos, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """PolarPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.vecs.CartesianPos1D)
    >>> x2
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("km")))
    >>> x2.x
    Quantity['length'](Array(0.9848077, dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * jnp.cos(current.phi))


@dispatch
def represent_as(
    current: PolarPos, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """PolarPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.vecs.RadialPos)
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("km")))
    >>> x2.r
    Distance(Array(1., dtype=float32), unit='km')

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: PolarPos,
    target: type[SphericalPos],
    /,
    theta: Quantity["angle"] = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> SphericalPos:
    """PolarPos -> SphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.SphericalPos, theta=u.Quantity(14, "deg"))
    >>> x2
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                  theta=Angle(value=f32[], unit=Unit("deg")),
                  phi=Angle(value=f32[], unit=Unit("deg")) )
    >>> x2.theta
    Angle(Array(14., dtype=float32), unit='deg')

    """
    return target.from_(r=current.r, theta=theta, phi=current.phi)


@dispatch
def represent_as(
    current: PolarPos,
    target: type[MathSphericalPos],
    /,
    phi: Quantity["angle"] = Quantity(0.0, "radian"),
    **kwargs: Any,
) -> MathSphericalPos:
    """PolarPos -> MathSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.vecs.MathSphericalPos, phi=u.Quantity(14, "deg"))
    >>> x2
    MathSphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                      theta=Angle(value=f32[], unit=Unit("deg")),
                      phi=Angle(value=f32[], unit=Unit("deg")) )
    >>> x2.phi
    Angle(Array(14., dtype=float32), unit='deg')

    """
    return target.from_(r=current.r, phi=phi, theta=current.phi)


@dispatch
def represent_as(
    current: PolarPos,
    target: type[CylindricalPos],
    /,
    *,
    z: Quantity["length"] = Quantity(0.0, "m"),
    **kwargs: Any,
) -> CylindricalPos:
    """PolarPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.PolarPos(r=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"))

    >>> x2 = cx.represent_as(x, cx.vecs.CylindricalPos, z=u.Quantity(14, "km"))
    >>> x2
    CylindricalPos( rho=Quantity[...](value=f32[], unit=Unit("km")),
                    phi=Angle(value=f32[], unit=Unit("deg")),
                    z=Quantity[...](value=f32[], unit=Unit("km")) )
    >>> x2.z
    Quantity['length'](Array(14., dtype=float32), unit='km')

    """
    return target(rho=current.r, phi=current.phi, z=z)
