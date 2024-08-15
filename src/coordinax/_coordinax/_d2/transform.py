"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.array_api as xp

from .base import AbstractPosition2D, AbstractVelocity2D
from .cartesian import CartesianAcceleration2D, CartesianPosition2D, CartesianVelocity2D
from .polar import PolarPosition, PolarVelocity
from coordinax._coordinax.base_pos import AbstractPosition


@dispatch
def represent_as(
    current: AbstractPosition2D, target: type[AbstractPosition2D], /, **kwargs: Any
) -> AbstractPosition2D:
    """AbstractPosition2D -> Cartesian2D -> AbstractPosition2D.

    This is the base case for the transformation of 2D vectors.
    """
    return represent_as(represent_as(current, CartesianPosition2D), target)


@dispatch.multi(
    (CartesianPosition2D, type[CartesianPosition2D]),
    (PolarPosition, type[PolarPosition]),
)
def represent_as(
    current: AbstractPosition2D, target: type[AbstractPosition2D], /, **kwargs: Any
) -> AbstractPosition2D:
    """Self transform of 2D vectors."""
    return current


@dispatch.multi(
    (CartesianVelocity2D, type[CartesianVelocity2D], AbstractPosition),
    (PolarVelocity, type[PolarVelocity], AbstractPosition),
)
def represent_as(
    current: AbstractVelocity2D,
    target: type[AbstractVelocity2D],
    position: AbstractPosition,
    /,
    **kwargs: Any,
) -> AbstractVelocity2D:
    """Self transform of 2D Differentials."""
    return current


# =============================================================================
# CartesianPosition2D


@dispatch
def represent_as(
    current: CartesianPosition2D, target: type[PolarPosition], /, **kwargs: Any
) -> PolarPosition:
    """CartesianPosition2D -> PolarPosition.

    The `x` and `y` coordinates are converted to the radial coordinate `r` and
    the angular coordinate `phi`.
    """
    r = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, phi=phi)


# =============================================================================
# PolarPosition


@dispatch
def represent_as(
    current: PolarPosition, target: type[CartesianPosition2D], /, **kwargs: Any
) -> CartesianPosition2D:
    """PolarPosition -> CartesianPosition2D."""
    x = current.r.distance * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.phi)
    return target(x=x, y=y)


# =============================================================================
# CartesianVelocity2D


@dispatch
def represent_as(
    current: CartesianVelocity2D, target: type[CartesianVelocity2D], /
) -> CartesianVelocity2D:
    """CartesianVelocity2D -> CartesianVelocity2D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVelocity2D.constructor([1, 1], "m/s")
    >>> cx.represent_as(v, cx.CartesianVelocity2D) is v
    True

    """
    return current


# =============================================================================
# CartesianAcceleration2D


@dispatch
def represent_as(
    current: CartesianAcceleration2D, target: type[CartesianAcceleration2D], /
) -> CartesianAcceleration2D:
    """CartesianAcceleration2D -> CartesianAcceleration2D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.CartesianAcceleration2D.constructor([1, 1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAcceleration2D) is a
    True

    """
    return current
