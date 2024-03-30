"""Transformations between representations."""

__all__ = ["represent_as"]

from typing import Any
from warnings import warn

from plum import dispatch

import quaxed.array_api as xp

from coordinax._d1.builtin import Cartesian1DVector, RadialVector
from coordinax._d2.base import Abstract2DVector
from coordinax._d2.builtin import Cartesian2DVector, PolarVector
from coordinax._d3.builtin import Cartesian3DVector, CylindricalVector
from coordinax._d3.sphere import MathSphericalVector, SphericalVector
from coordinax._exceptions import IrreversibleDimensionChange

# =============================================================================
# Cartesian3DVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """Cartesian3DVector -> Cartesian1DVector.

    The `y` and `z` coordinates are dropped.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian3DVector.constructor(Quantity([1.0, 2.0, 3.0], "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian1DVector)
    >>> x2
    Cartesian1DVector(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("km"))
    )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """Cartesian3DVector -> RadialVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian3DVector.constructor(Quantity([1.0, 2.0, 3.0], "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialVector)
    >>> x2
    RadialVector(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=xp.sqrt(current.x**2 + current.y**2 + current.z**2))


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """Cartesian3DVector -> Cartesian2DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian3DVector.constructor(Quantity([1.0, 2.0, 3.0], "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian2DVector)
    >>> x2
    Cartesian2DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x, y=current.y)


@dispatch.multi(
    (Cartesian3DVector, type[PolarVector]),
)
def represent_as(
    current: Cartesian3DVector, target: type[Abstract2DVector], /, **kwargs: Any
) -> Abstract2DVector:
    """Cartesian3DVector -> Cartesian2D -> Abstract2DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.Cartesian3DVector.constructor(Quantity([1.0, 2.0, 3.0], "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarVector)
    >>> x2
    PolarVector( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    cart2 = represent_as(current, Cartesian2DVector)
    return represent_as(cart2, target)


# =============================================================================
# CylindricalVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: CylindricalVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """CylindricalVector -> Cartesian1DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalVector(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian1DVector)
    >>> x2
    Cartesian1DVector( x=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.rho * xp.cos(current.phi))


@dispatch
def represent_as(
    current: CylindricalVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """CylindricalVector -> RadialVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalVector(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialVector)
    >>> x2
    RadialVector(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: CylindricalVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """CylindricalVector -> Cartesian2DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalVector(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian2DVector)
    >>> x2
    Cartesian2DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.rho * xp.cos(current.phi)
    y = current.rho * xp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: CylindricalVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """CylindricalVector -> PolarVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CylindricalVector(rho=Quantity(1.0, "km"), phi=Quantity(10.0, "deg"),
    ...                          z=Quantity(14, "km"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarVector)
    >>> x2
    PolarVector( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("deg")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho, phi=current.phi)


# =============================================================================
# SphericalVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: SphericalVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """SphericalVector -> Cartesian1DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalVector(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian1DVector)
    >>> x2
    Cartesian1DVector( x=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * xp.sin(current.theta) * xp.cos(current.phi))


@dispatch
def represent_as(
    current: SphericalVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """SphericalVector -> RadialVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalVector(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialVector)
    >>> x2
    RadialVector(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: SphericalVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """SphericalVector -> Cartesian2DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalVector(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian2DVector)
    >>> x2
    Cartesian2DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r * xp.sin(current.theta) * xp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: SphericalVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """SphericalVector -> PolarVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.SphericalVector(r=Quantity(1.0, "km"), theta=Quantity(14, "deg"),
    ...                        phi=Quantity(10.0, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarVector)
    >>> x2
    PolarVector( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("deg")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * xp.sin(current.theta), phi=current.phi)


# =============================================================================
# MathSphericalVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: MathSphericalVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """MathSphericalVector -> Cartesian1DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalVector(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian1DVector)
    >>> x2
    Cartesian1DVector( x=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * xp.sin(current.phi) * xp.cos(current.theta))


@dispatch
def represent_as(
    current: MathSphericalVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """MathSphericalVector -> RadialVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalVector(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.RadialVector)
    >>> x2
    RadialVector(r=Distance(value=f32[], unit=Unit("km")))

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: MathSphericalVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """MathSphericalVector -> Cartesian2DVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalVector(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.Cartesian2DVector)
    >>> x2
    Cartesian2DVector( x=Quantity[...](value=f32[], unit=Unit("km")),
                       y=Quantity[...](value=f32[], unit=Unit("km")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * xp.sin(current.phi) * xp.cos(current.theta)
    y = current.r * xp.sin(current.phi) * xp.sin(current.theta)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: MathSphericalVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """MathSphericalVector -> PolarVector.

    Examples
    --------
    >>> import warnings
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.MathSphericalVector(r=Quantity(1.0, "km"), theta=Quantity(10.0, "deg"),
    ...                            phi=Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.represent_as(x, cx.PolarVector)
    >>> x2
    PolarVector( r=Distance(value=f32[], unit=Unit("km")),
                 phi=Quantity[...](value=f32[], unit=Unit("deg")) )

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * xp.sin(current.phi), phi=current.theta)
