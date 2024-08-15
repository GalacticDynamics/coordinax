"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .base import AbstractAcceleration1D, AbstractPosition1D, AbstractVelocity1D
from .cartesian import CartesianAcceleration1D, CartesianPosition1D, CartesianVelocity1D
from .radial import RadialAcceleration, RadialPosition, RadialVelocity
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.base_vel import AbstractVelocity

###############################################################################
# 1D


@dispatch
def represent_as(
    current: AbstractPosition1D, target: type[AbstractPosition1D], /, **kwargs: Any
) -> AbstractPosition1D:
    """AbstractPosition1D -> Cartesian1D -> AbstractPosition1D.

    This is the base case for the transformation of 1D vectors.
    """
    cart1d = represent_as(current, CartesianPosition1D)
    return represent_as(cart1d, target)


# TODO: use multi, with precedence
@dispatch(precedence=1)
def represent_as(
    current: CartesianPosition1D, target: type[CartesianPosition1D], /, **kwargs: Any
) -> CartesianPosition1D:
    """Self transform of 1D vectors."""
    return current


# TODO: use multi, with precedence
@dispatch(precedence=1)
def represent_as(
    current: RadialPosition, target: type[RadialPosition], /, **kwargs: Any
) -> RadialPosition:
    """Self transform of 1D vectors."""
    return current


@dispatch.multi(
    (CartesianVelocity1D, type[CartesianVelocity1D], AbstractPosition),
    (RadialVelocity, type[RadialVelocity], AbstractPosition),
)
def represent_as(
    current: AbstractVelocity1D,
    target: type[AbstractVelocity1D],
    position: AbstractPosition,
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
        AbstractPosition,
    ),
    (RadialAcceleration, type[RadialAcceleration], AbstractVelocity, AbstractPosition),
)
def represent_as(
    current: AbstractAcceleration1D,
    target: type[AbstractAcceleration1D],
    velocity: AbstractVelocity,
    position: AbstractPosition,
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
# CartesianPosition1D


@dispatch
def represent_as(
    current: CartesianPosition1D, target: type[RadialPosition], /, **kwargs: Any
) -> RadialPosition:
    """CartesianPosition1D -> RadialPosition.

    The `x` coordinate is converted to the radial coordinate `r`.
    """
    return target(r=current.x)


# =============================================================================
# RadialPosition


@dispatch
def represent_as(
    current: RadialPosition, target: type[CartesianPosition1D], /, **kwargs: Any
) -> CartesianPosition1D:
    """RadialPosition -> CartesianPosition1D.

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
    >>> v = cx.CartesianVelocity1D.constructor([1], "m/s")
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
    >>> a = cx.CartesianAcceleration1D.constructor([1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAcceleration1D) is a
    True

    """
    return current
