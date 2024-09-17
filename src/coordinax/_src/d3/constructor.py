# pylint: disable=duplicate-code
"""3-dimensional constructors."""

__all__: list[str] = []

from typing import Any

from .base import AbstractAcceleration3D, AbstractPosition3D, AbstractVelocity3D
from .cartesian import CartesianAcceleration3D, CartesianPosition3D, CartesianVelocity3D

#####################################################################


@AbstractPosition3D.constructor._f.dispatch(precedence=-1)  # noqa: SLF001
def constructor(cls: type[AbstractPosition3D], obj: Any, /) -> CartesianPosition3D:
    """Try to construct a 3D Cartesian position from an object.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = Quantity([1, 2, 3], "km")
    >>> cx.AbstractPosition3D.constructor(x)
    CartesianPosition3D(
      x=Quantity[...](value=f32[], unit=Unit("km")),
      y=Quantity[...](value=f32[], unit=Unit("km")),
      z=Quantity[...](value=f32[], unit=Unit("km"))
    )

    """
    return (
        obj
        if isinstance(obj, CartesianPosition3D)
        else CartesianPosition3D.constructor(obj)
    )


@AbstractPosition3D.constructor._f.dispatch(precedence=1)  # noqa: SLF001
def constructor(
    cls: type[AbstractPosition3D], obj: AbstractPosition3D, /
) -> AbstractPosition3D:
    """Construct from a 3D position.

    Examples
    --------
    >>> import coordinax as cx

    >>> cart = cx.CartesianPosition3D.constructor([1, 2, 3], "km")
    >>> cx.AbstractPosition3D.constructor(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalPosition)
    >>> cx.AbstractPosition3D.constructor(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalPosition)
    >>> cx.AbstractPosition3D.constructor(cyl) is cyl
    True

    """
    return obj


#####################################################################


@AbstractVelocity3D.constructor._f.dispatch(precedence=-1)  # noqa: SLF001
def constructor(cls: type[AbstractVelocity3D], obj: Any, /) -> CartesianVelocity3D:
    """Try to construct a 3D Cartesian velocity from an object.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = Quantity([1, 2, 3], "km / s")
    >>> cx.AbstractVelocity3D.constructor(x)
    CartesianVelocity3D(
      d_x=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return (
        obj
        if isinstance(obj, CartesianVelocity3D)
        else CartesianVelocity3D.constructor(obj)
    )


@AbstractVelocity3D.constructor._f.dispatch(precedence=1)  # noqa: SLF001
def constructor(
    cls: type[AbstractVelocity3D], obj: AbstractVelocity3D, /
) -> AbstractVelocity3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPosition3D.constructor([1, 1, 1], "km")

    >>> cart = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
    >>> cx.AbstractVelocity3D.constructor(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalVelocity, q)
    >>> cx.AbstractVelocity3D.constructor(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalVelocity, q)
    >>> cx.AbstractVelocity3D.constructor(cyl) is cyl
    True

    """
    return obj


#####################################################################


@AbstractAcceleration3D.constructor._f.dispatch(precedence=-1)  # noqa: SLF001
def constructor(
    cls: type[AbstractAcceleration3D], obj: Any, /
) -> CartesianAcceleration3D:
    """Try to construct a 3D Cartesian velocity from an object.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = Quantity([1, 2, 3], "km / s2")
    >>> cx.AbstractAcceleration3D.constructor(x)
    CartesianAcceleration3D(
      d2_x=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_y=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_z=Quantity[...](value=f32[], unit=Unit("km / s2"))
    )

    """
    return (
        obj
        if isinstance(obj, CartesianAcceleration3D)
        else CartesianAcceleration3D.constructor(obj)
    )


@AbstractAcceleration3D.constructor._f.dispatch(precedence=1)  # noqa: SLF001
def constructor(
    cls: type[AbstractAcceleration3D], obj: AbstractAcceleration3D, /
) -> AbstractAcceleration3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPosition3D.constructor([1, 1, 1], "km")
    >>> p = cx.CartesianVelocity3D.constructor([1, 1, 1], "km/s")

    >>> cart = cx.CartesianAcceleration3D.constructor([1, 2, 3], "km/s2")
    >>> cx.AbstractAcceleration3D.constructor(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalAcceleration, p, q)
    >>> cx.AbstractAcceleration3D.constructor(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalAcceleration, p, q)
    >>> cx.AbstractAcceleration3D.constructor(cyl) is cyl
    True

    """
    return obj
