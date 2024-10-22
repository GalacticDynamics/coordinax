# pylint: disable=duplicate-code
"""3-dimensional constructors."""

__all__: list[str] = []

from typing import Any

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D

#####################################################################


@AbstractPos3D.from_._f.dispatch(precedence=-1)  # noqa: SLF001
def from_(cls: type[AbstractPos3D], obj: Any, /) -> CartesianPos3D:
    """Try to construct a 3D Cartesian position from an object.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = Quantity([1, 2, 3], "km")
    >>> cx.AbstractPos3D.from_(x)
    CartesianPos3D(
      x=Quantity[...](value=f32[], unit=Unit("km")),
      y=Quantity[...](value=f32[], unit=Unit("km")),
      z=Quantity[...](value=f32[], unit=Unit("km"))
    )

    """
    return obj if isinstance(obj, CartesianPos3D) else CartesianPos3D.from_(obj)


@AbstractPos3D.from_._f.dispatch(precedence=1)  # noqa: SLF001
def from_(cls: type[AbstractPos3D], obj: AbstractPos3D, /) -> AbstractPos3D:
    """Construct from a 3D position.

    Examples
    --------
    >>> import coordinax as cx

    >>> cart = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> cx.AbstractPos3D.from_(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalPos)
    >>> cx.AbstractPos3D.from_(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalPos)
    >>> cx.AbstractPos3D.from_(cyl) is cyl
    True

    """
    return obj


#####################################################################


@AbstractVel3D.from_._f.dispatch(precedence=-1)  # noqa: SLF001
def from_(cls: type[AbstractVel3D], obj: Any, /) -> CartesianVel3D:
    """Try to construct a 3D Cartesian velocity from an object.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = Quantity([1, 2, 3], "km / s")
    >>> cx.AbstractVel3D.from_(x)
    CartesianVel3D(
      d_x=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    """
    return obj if isinstance(obj, CartesianVel3D) else CartesianVel3D.from_(obj)


@AbstractVel3D.from_._f.dispatch(precedence=1)  # noqa: SLF001
def from_(cls: type[AbstractVel3D], obj: AbstractVel3D, /) -> AbstractVel3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 1, 1], "km")

    >>> cart = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cx.AbstractVel3D.from_(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalVel, q)
    >>> cx.AbstractVel3D.from_(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalVel, q)
    >>> cx.AbstractVel3D.from_(cyl) is cyl
    True

    """
    return obj


#####################################################################


@AbstractAcc3D.from_._f.dispatch(precedence=-1)  # noqa: SLF001
def from_(cls: type[AbstractAcc3D], obj: Any, /) -> CartesianAcc3D:
    """Try to construct a 3D Cartesian velocity from an object.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = Quantity([1, 2, 3], "km / s2")
    >>> cx.AbstractAcc3D.from_(x)
    CartesianAcc3D(
      d2_x=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_y=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_z=Quantity[...](value=f32[], unit=Unit("km / s2"))
    )

    """
    return obj if isinstance(obj, CartesianAcc3D) else CartesianAcc3D.from_(obj)


@AbstractAcc3D.from_._f.dispatch(precedence=1)  # noqa: SLF001
def from_(cls: type[AbstractAcc3D], obj: AbstractAcc3D, /) -> AbstractAcc3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 1, 1], "km")
    >>> p = cx.CartesianVel3D.from_([1, 1, 1], "km/s")

    >>> cart = cx.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> cx.AbstractAcc3D.from_(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalAcc, p, q)
    >>> cx.AbstractAcc3D.from_(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalAcc, p, q)
    >>> cx.AbstractAcc3D.from_(cyl) is cyl
    True

    """
    return obj
