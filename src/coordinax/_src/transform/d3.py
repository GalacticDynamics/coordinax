"""Transformations between representations."""

__all__: list[str] = []

from typing import Any
from warnings import warn

from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.d1.cartesian import CartesianPos1D
from coordinax._src.d1.radial import RadialPos
from coordinax._src.d2.base import AbstractPos2D
from coordinax._src.d2.cartesian import CartesianPos2D
from coordinax._src.d2.polar import PolarPos
from coordinax._src.d3.cartesian import CartesianPos3D
from coordinax._src.d3.cylindrical import CylindricalPos
from coordinax._src.d3.mathspherical import MathSphericalPos
from coordinax._src.d3.spherical import SphericalPos
from coordinax._src.exceptions import IrreversibleDimensionChange

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
    >>> from unxt import Quantity
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
    >>> from unxt import Quantity
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
    >>> from unxt import Quantity
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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalPos(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("deg")) )

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalPos(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("deg")) )

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

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
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalPos(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarPos)
    >>> x2
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("deg")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * jnp.sin(current.phi), phi=current.theta)
