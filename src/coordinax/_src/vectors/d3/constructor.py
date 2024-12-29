"""3-dimensional constructors."""
# pylint: disable=duplicate-code

__all__: list[str] = []

from plum import dispatch

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D

#####################################################################


@dispatch(precedence=1)
def vector(cls: type[AbstractPos3D], obj: AbstractPos3D, /) -> AbstractPos3D:
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


@dispatch(precedence=1)
def vector(cls: type[AbstractVel3D], obj: AbstractVel3D, /) -> AbstractVel3D:
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


@dispatch(precedence=1)
def vector(cls: type[AbstractAcc3D], obj: AbstractAcc3D, /) -> AbstractAcc3D:
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
