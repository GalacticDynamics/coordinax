"""Transformations between representations."""

__all__: list[str] = []

from typing import Any
from warnings import warn

from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.vectors.d1 import CartesianPos1D, RadialPos
from coordinax._src.vectors.d2 import AbstractPos2D, CartesianPos2D, PolarPos
from coordinax._src.vectors.d3 import (
    CartesianPos3D,
    CylindricalPos,
    MathSphericalPos,
    ProlateSpheroidalPos,
    SphericalPos,
)
from coordinax._src.vectors.exceptions import IrreversibleDimensionChange

# =============================================================================
# CartesianPos3D


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: CartesianPos3D, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """CartesianPos3D -> CartesianPos1D.

    The `y` and `z` coordinates are dropped.

    Examples
    --------
    >>> import warnings
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos1D)
    >>> x2
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def represent_as(
    current: CartesianPos3D, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """CartesianPos3D -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialPos)
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=jnp.sqrt(current.x**2 + current.y**2 + current.z**2))


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: CartesianPos3D, target: type[CartesianPos2D], /, **kwargs: Any
) -> CartesianPos2D:
    """CartesianPos3D -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos2D)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x, y=current.y)


@dispatch.multi(
    (CartesianPos3D, type[PolarPos]),
)
def represent_as(
    current: CartesianPos3D, target: type[AbstractPos2D], /, **kwargs: Any
) -> AbstractPos2D:
    """CartesianPos3D -> Cartesian2D -> AbstractPos2D.

    Examples
    --------
    >>> import warnings
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("rad")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    cart2 = represent_as(current, CartesianPos2D)
    return represent_as(cart2, target)


# =============================================================================
# CylindricalPos


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: CylindricalPos, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """CylindricalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"),
    ...                       z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos1D)
    >>> x2
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.rho * jnp.cos(current.phi))


@dispatch
def represent_as(
    current: CylindricalPos, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """CylindricalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"),
    ...                        z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialPos)
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: CylindricalPos, target: type[CartesianPos2D], /, **kwargs: Any
) -> CartesianPos2D:
    """CylindricalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"),
    ...                       z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos2D)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.rho * jnp.cos(current.phi)
    y = current.rho * jnp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: CylindricalPos, target: type[PolarPos], /, **kwargs: Any
) -> PolarPos:
    """CylindricalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=u.Quantity(1.0, "km"), phi=u.Quantity(10.0, "deg"),
    ...                       z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("deg")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho, phi=current.phi)


# =============================================================================
# SphericalPos


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: SphericalPos, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """SphericalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos1D)
    >>> x2
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * jnp.sin(current.theta) * jnp.cos(current.phi))


@dispatch
def represent_as(
    current: SphericalPos, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """SphericalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialPos)
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: SphericalPos, target: type[CartesianPos2D], /, **kwargs: Any
) -> CartesianPos2D:
    """SphericalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos2D)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * jnp.sin(current.theta) * jnp.cos(current.phi)
    y = current.r * jnp.sin(current.theta) * jnp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: SphericalPos, target: type[PolarPos], /, **kwargs: Any
) -> PolarPos:
    """SphericalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("deg")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * jnp.sin(current.theta), phi=current.phi)


# =============================================================================
# MathSphericalPos


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: MathSphericalPos, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """MathSphericalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(10.0, "deg"),
    ...                         phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos1D)
    >>> x2
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * jnp.sin(current.phi) * jnp.cos(current.theta))


@dispatch
def represent_as(
    current: MathSphericalPos, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """MathSphericalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(10.0, "deg"),
    ...                         phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialPos)
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: MathSphericalPos, target: type[CartesianPos2D], /, **kwargs: Any
) -> CartesianPos2D:
    """MathSphericalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(10.0, "deg"),
    ...                         phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos2D)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * jnp.sin(current.phi) * jnp.cos(current.theta)
    y = current.r * jnp.sin(current.phi) * jnp.sin(current.theta)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: MathSphericalPos, target: type[PolarPos], /, **kwargs: Any
) -> PolarPos:
    """MathSphericalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=u.Quantity(1.0, "km"), theta=u.Quantity(10.0, "deg"),
    ...                         phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("deg")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * jnp.sin(current.phi), phi=current.theta)


# =============================================================================
# ProlateSpheroidalPos


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: ProlateSpheroidalPos, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """ProlateSpheroidalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(2.0, "kpc2"),
    ...     nu=Quantity(0.5, "kpc2"),
    ...     phi=Quantity(0.5, "rad"),
    ...     Delta=Quantity(1.0, "kpc"),
    ... )

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos1D)
    >>> x2
    CartesianPos1D(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
    )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return represent_as(represent_as(current, CartesianPos3D), target)


@dispatch
def represent_as(
    current: ProlateSpheroidalPos, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """ProlateSpheroidalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(2.0, "kpc2"),
    ...     nu=Quantity(0.5, "kpc2"),
    ...     phi=Quantity(0.5, "rad"),
    ...     Delta=Quantity(1.0, "kpc"),
    ... )

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialPos)
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("kpc")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return represent_as(represent_as(current, CartesianPos3D), target)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: ProlateSpheroidalPos, target: type[CartesianPos2D], /, **kwargs: Any
) -> CartesianPos2D:
    """ProlateSpheroidalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(2.0, "kpc2"),
    ...     nu=Quantity(0.5, "kpc2"),
    ...     phi=Quantity(0.5, "rad"),
    ...     Delta=Quantity(1.0, "kpc"),
    ... )

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.CartesianPos2D)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("kpc")),
                       y=Quantity[...](value=f32[], unit=Unit("kpc")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return represent_as(represent_as(current, CartesianPos3D), target)


@dispatch
def represent_as(
    current: ProlateSpheroidalPos, target: type[PolarPos], /, **kwargs: Any
) -> PolarPos:
    """ProlateSpheroidalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(2.0, "kpc2"),
    ...     nu=Quantity(0.5, "kpc2"),
    ...     phi=Quantity(0.5, "rad"),
    ...     Delta=Quantity(1.0, "kpc"),
    ... )

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("kpc")),
              phi=Angle(value=f32[], unit=Unit("rad")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return represent_as(represent_as(current, CartesianPos3D), target)
