"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from .cartesian import CartesianAcc1D, CartesianPos1D, CartesianVel1D
from .radial import RadialAcc, RadialPos, RadialVel
from coordinax._src.vectors.base import AbstractPos, AbstractVel

###############################################################################
# 1D


@dispatch
def vconvert(
    target: type[AbstractPos1D], current: AbstractPos1D, /, **kwargs: Any
) -> AbstractPos1D:
    """AbstractPos1D -> Cartesian1D -> AbstractPos1D.

    This is the base case for the transformation of 1D vectors.
    """
    cart1d = vconvert(CartesianPos1D, current)
    return vconvert(target, cart1d)


# TODO: use multi, with precedence
@dispatch(precedence=1)
def vconvert(
    target: type[CartesianPos1D], current: CartesianPos1D, /, **kwargs: Any
) -> CartesianPos1D:
    """Self transform of 1D vectors."""
    return current


# TODO: use multi, with precedence
@dispatch(precedence=1)
def vconvert(
    target: type[RadialPos], current: RadialPos, /, **kwargs: Any
) -> RadialPos:
    """Self transform of 1D vectors."""
    return current


@dispatch.multi(
    (type[CartesianVel1D], CartesianVel1D, AbstractPos),
    (type[RadialVel], RadialVel, AbstractPos),
)
def vconvert(
    target: type[AbstractVel1D],
    current: AbstractVel1D,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVel1D:
    """Self transform of 1D Velocities."""
    return current


# Special-case where self-transform doesn't need position
@dispatch.multi(
    (type[CartesianVel1D], CartesianVel1D),
    (type[RadialVel], RadialVel),
)
def vconvert(
    target: type[AbstractVel1D], current: AbstractVel1D, /, **kwargs: Any
) -> AbstractVel1D:
    """Self transform of 1D Velocities."""
    return current


@dispatch.multi(
    (
        type[CartesianAcc1D],
        CartesianAcc1D,
        AbstractVel,
        AbstractPos,
    ),
    (type[RadialAcc], RadialAcc, AbstractVel, AbstractPos),
)
def vconvert(
    target: type[AbstractAcc1D],
    current: AbstractAcc1D,
    velocity: AbstractVel,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractAcc1D:
    """Self transform of 1D Accs."""
    return current


# Special-case where self-transform doesn't need velocity or position
@dispatch.multi(
    (type[CartesianAcc1D], CartesianAcc1D),
    (type[RadialAcc], RadialAcc),
)
def vconvert(
    target: type[AbstractAcc1D],
    current: AbstractAcc1D,
    /,
    **kwargs: Any,
) -> AbstractAcc1D:
    """Self transform of 1D Accs."""
    return current


# =============================================================================
# CartesianPos1D


@dispatch
def vconvert(
    target: type[RadialPos], current: CartesianPos1D, /, **kwargs: Any
) -> RadialPos:
    """CartesianPos1D -> RadialPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    """
    return target(r=current.x)


# =============================================================================
# RadialPos


@dispatch
def vconvert(
    target: type[CartesianPos1D], current: RadialPos, /, **kwargs: Any
) -> CartesianPos1D:
    """RadialPos -> CartesianPos1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.
    """
    return target(x=current.r.distance)


# =============================================================================
# CartesianVel1D


@dispatch
def vconvert(
    target: type[CartesianVel1D], current: CartesianVel1D, /
) -> CartesianVel1D:
    """CartesianVel1D -> CartesianVel1D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.vecs.CartesianVel1D.from_([1], "m/s")
    >>> cx.vconvert(cx.vecs.CartesianVel1D, v) is v
    True

    """
    return current


# =============================================================================
# CartesianAcc1D


@dispatch
def vconvert(
    target: type[CartesianAcc1D], current: CartesianAcc1D, /
) -> CartesianAcc1D:
    """CartesianAcc1D -> CartesianAcc1D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.vecs.CartesianAcc1D.from_([1], "m/s2")
    >>> cx.vconvert(cx.vecs.CartesianAcc1D, a) is a
    True

    """
    return current
