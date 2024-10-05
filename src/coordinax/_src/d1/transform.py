"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .base import AbstractAcceleration1D, AbstractPos1D, AbstractVelocity1D
from .cartesian import CartesianAcceleration1D, CartesianPos1D, CartesianVelocity1D
from .radial import RadialAcceleration, RadialPos, RadialVelocity
from coordinax._src.base import AbstractPos, AbstractVelocity

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
    (CartesianVelocity1D, type[CartesianVelocity1D], AbstractPos),
    (RadialVelocity, type[RadialVelocity], AbstractPos),
)
def represent_as(
    current: AbstractVelocity1D,
    target: type[AbstractVelocity1D],
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVelocity1D:
    """Self transform of 1D Velocities."""
    return current


# Special-case where self-transform doesn't need position
@dispatch.multi(
    (CartesianVelocity1D, type[CartesianVelocity1D]),
    (RadialVelocity, type[RadialVelocity]),
)
def represent_as(
    current: AbstractVelocity1D, target: type[AbstractVelocity1D], /, **kwargs: Any
) -> AbstractVelocity1D:
    """Self transform of 1D Velocities."""
    return current


@dispatch.multi(
    (
        CartesianAcceleration1D,
        type[CartesianAcceleration1D],
        AbstractVelocity,
        AbstractPos,
    ),
    (RadialAcceleration, type[RadialAcceleration], AbstractVelocity, AbstractPos),
)
def represent_as(
    current: AbstractAcceleration1D,
    target: type[AbstractAcceleration1D],
    velocity: AbstractVelocity,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractAcceleration1D:
    """Self transform of 1D Accelerations."""
    return current


# Special-case where self-transform doesn't need velocity or position
@dispatch.multi(
    (CartesianAcceleration1D, type[CartesianAcceleration1D]),
    (RadialAcceleration, type[RadialAcceleration]),
)
def represent_as(
    current: AbstractAcceleration1D,
    target: type[AbstractAcceleration1D],
    /,
    **kwargs: Any,
) -> AbstractAcceleration1D:
    """Self transform of 1D Accelerations."""
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
# CartesianVelocity1D


@dispatch
def represent_as(
    current: CartesianVelocity1D, target: type[CartesianVelocity1D], /
) -> CartesianVelocity1D:
    """CartesianVelocity1D -> CartesianVelocity1D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVelocity1D.from_([1], "m/s")
    >>> cx.represent_as(v, cx.CartesianVelocity1D) is v
    True

    """
    return current


# =============================================================================
# CartesianAcceleration1D


@dispatch
def represent_as(
    current: CartesianAcceleration1D, target: type[CartesianAcceleration1D], /
) -> CartesianAcceleration1D:
    """CartesianAcceleration1D -> CartesianAcceleration1D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.CartesianAcceleration1D.from_([1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAcceleration1D) is a
    True

    """
    return current
