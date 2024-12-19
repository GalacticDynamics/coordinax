"""3-dimensional constructors."""
# pylint: disable=duplicate-code

__all__: list[str] = []

from typing import Any

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D

#####################################################################


@AbstractPos3D.from_.dispatch(precedence=-1)
def from_(cls: type[AbstractPos3D], obj: Any, /) -> CartesianPos3D:
    """Try to construct a 3D Cartesian position from an object.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = u.Quantity([1, 2, 3], "km")
    >>> vec = cx.vecs.AbstractPos3D.from_(x)
    >>> print(vec)
    <CartesianPos3D (x[km], y[km], z[km])
        [1 2 3]>

    """
    return obj if isinstance(obj, CartesianPos3D) else CartesianPos3D.from_(obj)


@AbstractPos3D.from_.dispatch(precedence=1)
def from_(cls: type[AbstractPos3D], obj: AbstractPos3D, /) -> AbstractPos3D:
    """Construct from a 3D position.

    Examples
    --------
    >>> import coordinax as cx

    >>> cart = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> cx.vecs.AbstractPos3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalPos)
    >>> cx.vecs.AbstractPos3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalPos)
    >>> cx.vecs.AbstractPos3D.from_(cyl) is cyl
    True

    """
    return obj


#####################################################################


@AbstractVel3D.from_.dispatch(precedence=-1)
def from_(cls: type[AbstractVel3D], obj: Any, /) -> CartesianVel3D:
    """Try to construct a 3D Cartesian velocity from an object.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = u.Quantity([1, 2, 3], "km / s")
    >>> vec = cx.vecs.AbstractVel3D.from_(x)
    >>> print(vec)
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [1 2 3]>

    """
    return obj if isinstance(obj, CartesianVel3D) else CartesianVel3D.from_(obj)


@AbstractVel3D.from_.dispatch(precedence=1)
def from_(cls: type[AbstractVel3D], obj: AbstractVel3D, /) -> AbstractVel3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 1, 1], "km")

    >>> cart = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cx.vecs.AbstractVel3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(cyl) is cyl
    True

    """
    return obj


#####################################################################


@AbstractAcc3D.from_.dispatch(precedence=-1)
def from_(cls: type[AbstractAcc3D], obj: Any, /) -> CartesianAcc3D:
    """Try to construct a 3D Cartesian velocity from an object.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = u.Quantity([1, 2, 3], "km / s2")
    >>> vec = cx.vecs.AbstractAcc3D.from_(x)
    >>> print(vec)
    <CartesianAcc3D (d2_x[km / s2], d2_y[km / s2], d2_z[km / s2])
        [1 2 3]>

    """
    return obj if isinstance(obj, CartesianAcc3D) else CartesianAcc3D.from_(obj)


@AbstractAcc3D.from_.dispatch(precedence=1)
def from_(cls: type[AbstractAcc3D], obj: AbstractAcc3D, /) -> AbstractAcc3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 1, 1], "km")
    >>> p = cx.CartesianVel3D.from_([1, 1, 1], "km/s")

    >>> cart = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> cx.vecs.AbstractAcc3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.vecs.SphericalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(cyl) is cyl
    True

    """
    return obj
