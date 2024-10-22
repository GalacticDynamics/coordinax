"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from .cartesian import CartesianAcc1D, CartesianPos1D, CartesianVel1D
from .radial import RadialAcc, RadialPos, RadialVel
from coordinax._src.base import AbstractPos, AbstractVel

###############################################################################
# 1D


@dispatch
def represent_as(
    current: AbstractPos1D, target: type[AbstractPos1D], /, **kwargs: Any
) -> AbstractPos1D:
    """AbstractPos1D -> Cartesian1D -> AbstractPos1D.

    This is the base case for the transformation of 1D vectors.
    """
    cart1d = represent_as(current, CartesianPos1D)
    return represent_as(cart1d, target)


# TODO: use multi, with precedence
@dispatch(precedence=1)
def represent_as(
    current: CartesianPos1D, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """Self transform of 1D vectors."""
    return current


# TODO: use multi, with precedence
@dispatch(precedence=1)
def represent_as(
    current: RadialPos, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """Self transform of 1D vectors."""
    return current


@dispatch.multi(
    (CartesianVel1D, type[CartesianVel1D], AbstractPos),
    (RadialVel, type[RadialVel], AbstractPos),
)
def represent_as(
    current: AbstractVel1D,
    target: type[AbstractVel1D],
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVel1D:
    """Self transform of 1D Velocities."""
    return current


# Special-case where self-transform doesn't need position
@dispatch.multi(
    (CartesianVel1D, type[CartesianVel1D]),
    (RadialVel, type[RadialVel]),
)
def represent_as(
    current: AbstractVel1D, target: type[AbstractVel1D], /, **kwargs: Any
) -> AbstractVel1D:
    """Self transform of 1D Velocities."""
    return current


@dispatch.multi(
    (
        CartesianAcc1D,
        type[CartesianAcc1D],
        AbstractVel,
        AbstractPos,
    ),
    (RadialAcc, type[RadialAcc], AbstractVel, AbstractPos),
)
def represent_as(
    current: AbstractAcc1D,
    target: type[AbstractAcc1D],
    velocity: AbstractVel,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractAcc1D:
    """Self transform of 1D Accs."""
    return current


# Special-case where self-transform doesn't need velocity or position
@dispatch.multi(
    (CartesianAcc1D, type[CartesianAcc1D]),
    (RadialAcc, type[RadialAcc]),
)
def represent_as(
    current: AbstractAcc1D,
    target: type[AbstractAcc1D],
    /,
    **kwargs: Any,
) -> AbstractAcc1D:
    """Self transform of 1D Accs."""
    return current


# =============================================================================
# CartesianPos1D


@dispatch
def represent_as(
    current: CartesianPos1D, target: type[RadialPos], /, **kwargs: Any
) -> RadialPos:
    """CartesianPos1D -> RadialPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    """
    return target(r=current.x)


# =============================================================================
# RadialPos


@dispatch
def represent_as(
    current: RadialPos, target: type[CartesianPos1D], /, **kwargs: Any
) -> CartesianPos1D:
    """RadialPos -> CartesianPos1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.
    """
    return target(x=current.r.distance)


# =============================================================================
# CartesianVel1D


@dispatch
def represent_as(
    current: CartesianVel1D, target: type[CartesianVel1D], /
) -> CartesianVel1D:
    """CartesianVel1D -> CartesianVel1D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVel1D.from_([1], "m/s")
    >>> cx.represent_as(v, cx.CartesianVel1D) is v
    True

    """
    return current


# =============================================================================
# CartesianAcc1D


@dispatch
def represent_as(
    current: CartesianAcc1D, target: type[CartesianAcc1D], /
) -> CartesianAcc1D:
    """CartesianAcc1D -> CartesianAcc1D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.CartesianAcc1D.from_([1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAcc1D) is a
    True

    """
    return current
