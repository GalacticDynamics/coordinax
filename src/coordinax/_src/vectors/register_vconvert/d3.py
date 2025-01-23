"""Transformations between representations."""

__all__: list[str] = []

from typing import Any
from warnings import warn

from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.vectors.d1 import AbstractPos1D, CartesianPos1D, RadialPos
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
def vconvert(
    target: type[CartesianPos1D], current: CartesianPos3D, /, **kwargs: Any
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
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> x2
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def vconvert(
    target: type[RadialPos], current: CartesianPos3D, /, **kwargs: Any
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
    ...     x2 = cx.vconvert(cx.vecs.RadialPos, x)
    >>> x2
    RadialPos(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=jnp.sqrt(current.x**2 + current.y**2 + current.z**2))


# -----------------------------------------------
# 2D


@dispatch
def vconvert(
    target: type[CartesianPos2D], current: CartesianPos3D, /, **kwargs: Any
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
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> x2
    CartesianPos2D( x=Quantity[...](value=f32[], unit=Unit("km")),
                    y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x, y=current.y)


@dispatch.multi(
    (type[PolarPos], CartesianPos3D),
)
def vconvert(
    target: type[AbstractPos2D], current: CartesianPos3D, /, **kwargs: Any
) -> AbstractPos2D:
    """CartesianPos3D -> Cartesian2D -> AbstractPos2D.

    Examples
    --------
    >>> import warnings
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("rad")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    cart2 = vconvert(CartesianPos2D, current)
    return vconvert(target, cart2)


# =============================================================================
# CylindricalPos


# -----------------------------------------------
# 1D


@dispatch
def vconvert(
    target: type[CartesianPos1D], current: CylindricalPos, /, **kwargs: Any
) -> CartesianPos1D:
    """CylindricalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CylindricalPos(rho=u.Quantity(1.0, "km"),
    ...                            phi=u.Quantity(10.0, "deg"),
    ...                            z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(x2)
    <CartesianPos1D (x[km])
        [0.985]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.rho * jnp.cos(current.phi))


@dispatch
def vconvert(
    target: type[RadialPos], current: CylindricalPos, /, **kwargs: Any
) -> RadialPos:
    """CylindricalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                            phi=u.Quantity(10, "deg"),
    ...                            z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.RadialPos, x)
    >>> print(x2)
    <RadialPos (r[km])
        [1]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho)


# -----------------------------------------------
# 2D


@dispatch
def vconvert(
    target: type[CartesianPos2D], current: CylindricalPos, /, **kwargs: Any
) -> CartesianPos2D:
    """CylindricalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                            phi=u.Quantity(10, "deg"),
    ...                            z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D (x[km], y[km])
        [0.985 0.174]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.rho * jnp.cos(current.phi)
    y = current.rho * jnp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def vconvert(
    target: type[PolarPos], current: CylindricalPos, /, **kwargs: Any
) -> PolarPos:
    """CylindricalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                            phi=u.Quantity(10, "deg"),
    ...                            z=u.Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(x2)
    <PolarPos (r[km], phi[deg])
        [ 1 10]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho, phi=current.phi)


# =============================================================================
# SphericalPos


# -----------------------------------------------
# 1D


@dispatch
def vconvert(
    target: type[CartesianPos1D], current: SphericalPos, /, **kwargs: Any
) -> CartesianPos1D:
    """SphericalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(x2)
    <CartesianPos1D (x[km])
        [0.238]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * jnp.sin(current.theta) * jnp.cos(current.phi))


@dispatch
def vconvert(
    target: type[RadialPos], current: SphericalPos, /, **kwargs: Any
) -> RadialPos:
    """SphericalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))

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
# 2D


@dispatch
def vconvert(
    target: type[CartesianPos2D], current: SphericalPos, /, **kwargs: Any
) -> CartesianPos2D:
    """SphericalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D (x[km], y[km])
        [0.238 0.042]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * jnp.sin(current.theta) * jnp.cos(current.phi)
    y = current.r * jnp.sin(current.theta) * jnp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def vconvert(
    target: type[PolarPos], current: SphericalPos, /, **kwargs: Any
) -> PolarPos:
    """SphericalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(x2)
    <PolarPos (r[km], phi[deg])
        [ 0.242 10.   ]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * jnp.sin(current.theta), phi=current.phi)


# =============================================================================
# MathSphericalPos


# -----------------------------------------------
# 1D


@dispatch
def vconvert(
    target: type[CartesianPos1D], current: MathSphericalPos, /, **kwargs: Any
) -> CartesianPos1D:
    """MathSphericalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                              theta=u.Quantity(10, "deg"),
    ...                              phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(x2)
    <CartesianPos1D (x[km])
        [0.238]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * jnp.sin(current.phi) * jnp.cos(current.theta))


@dispatch
def vconvert(
    target: type[RadialPos], current: MathSphericalPos, /, **kwargs: Any
) -> RadialPos:
    """MathSphericalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                              theta=u.Quantity(10, "deg"),
    ...                              phi=u.Quantity(14, "deg"))

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
# 2D


@dispatch
def vconvert(
    target: type[CartesianPos2D], current: MathSphericalPos, /, **kwargs: Any
) -> CartesianPos2D:
    """MathSphericalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                              theta=u.Quantity(10, "deg"),
    ...                              phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D (x[km], y[km])
        [0.238 0.042]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * jnp.sin(current.phi) * jnp.cos(current.theta)
    y = current.r * jnp.sin(current.phi) * jnp.sin(current.theta)
    return target(x=x, y=y)


@dispatch
def vconvert(
    target: type[PolarPos], current: MathSphericalPos, /, **kwargs: Any
) -> PolarPos:
    """MathSphericalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                              theta=u.Quantity(10, "deg"),
    ...                              phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(x2)
    <PolarPos (r[km], phi[deg])
        [ 0.242 10.   ]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * jnp.sin(current.phi), phi=current.theta)


# =============================================================================
# ProlateSpheroidalPos


# -----------------------------------------------
# 1D


@dispatch
def vconvert(
    target: type[AbstractPos1D], current: ProlateSpheroidalPos, /, **kwargs: Any
) -> AbstractPos1D:
    """ProlateSpheroidalPos -> AbstractPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(2, "km2"),
    ...     nu=u.Quantity(0.5, "km2"),
    ...     phi=u.Quantity(0.5, "rad"),
    ...     Delta=u.Quantity(1, "km"),
    ... )

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(x2)
    <CartesianPos1D (x[km])
        [0.621]>

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.RadialPos, x)
    >>> print(x2)
    <RadialPos (r[km])
        [1.225]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return current.vconvert(CartesianPos3D).vconvert(target)


# -----------------------------------------------
# 2D


@dispatch
def vconvert(
    target: type[AbstractPos2D], current: ProlateSpheroidalPos, /, **kwargs: Any
) -> AbstractPos2D:
    """ProlateSpheroidalPos -> AbstractPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(2, "km2"),
    ...     nu=u.Quantity(0.5, "km2"),
    ...     phi=u.Quantity(0.5, "rad"),
    ...     Delta=u.Quantity(1, "km"),
    ... )

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D (x[km], y[km])
        [0.621 0.339]>

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(x2)
    <PolarPos (r[km], phi[rad])
        [0.707 0.5  ]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return current.vconvert(CartesianPos3D).vconvert(target)
