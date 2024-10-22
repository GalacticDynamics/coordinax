"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.numpy as xp

from .base import AbstractPos2D, AbstractVel2D
from .cartesian import CartesianAcc2D, CartesianPos2D, CartesianVel2D
from .polar import PolarPos, PolarVel
from coordinax._src.base import AbstractPos


@dispatch
def represent_as(
    current: AbstractPos2D, target: type[AbstractPos2D], /, **kwargs: Any
) -> AbstractPos2D:
    """AbstractPos2D -> Cartesian2D -> AbstractPos2D.

    This is the base case for the transformation of 2D vectors.
    """
    return represent_as(represent_as(current, CartesianPos2D), target)


@dispatch.multi(
    (CartesianPos2D, type[CartesianPos2D]),
    (PolarPos, type[PolarPos]),
)
def represent_as(
    current: AbstractPos2D, target: type[AbstractPos2D], /, **kwargs: Any
) -> AbstractPos2D:
    """Self transform of 2D vectors."""
    return current


@dispatch.multi(
    (CartesianVel2D, type[CartesianVel2D], AbstractPos),
    (PolarVel, type[PolarVel], AbstractPos),
)
def represent_as(
    current: AbstractVel2D,
    target: type[AbstractVel2D],
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVel2D:
    """Self transform of 2D Differentials."""
    return current


# =============================================================================
# CartesianPos2D


@dispatch
def represent_as(
    current: CartesianPos2D, target: type[PolarPos], /, **kwargs: Any
) -> PolarPos:
    """CartesianPos2D -> PolarPos.

    The `x` and `y` coordinates are converted to the radial coordinate `r` and
    the angular coordinate `phi`.
    """
    r = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, phi=phi)


# =============================================================================
# PolarPos


@dispatch
def represent_as(
    current: PolarPos, target: type[CartesianPos2D], /, **kwargs: Any
) -> CartesianPos2D:
    """PolarPos -> CartesianPos2D."""
    x = current.r.distance * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.phi)
    return target(x=x, y=y)


# =============================================================================
# CartesianVel2D


@dispatch
def represent_as(
    current: CartesianVel2D, target: type[CartesianVel2D], /
) -> CartesianVel2D:
    """CartesianVel2D -> CartesianVel2D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVel2D.from_([1, 1], "m/s")
    >>> cx.represent_as(v, cx.CartesianVel2D) is v
    True

    """
    return current


# =============================================================================
# CartesianAcc2D


@dispatch
def represent_as(
    current: CartesianAcc2D, target: type[CartesianAcc2D], /
) -> CartesianAcc2D:
    """CartesianAcc2D -> CartesianAcc2D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.CartesianAcc2D.from_([1, 1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAcc2D) is a
    True

    """
    return current
