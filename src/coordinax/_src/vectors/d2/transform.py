"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.numpy as xp

from .base import AbstractPos2D, AbstractVel2D
from .cartesian import CartesianAcc2D, CartesianPos2D, CartesianVel2D
from .polar import PolarPos, PolarVel
from coordinax._src.vectors.base import AbstractPos


@dispatch
def vconvert(
    current: AbstractPos2D, target: type[AbstractPos2D], /, **kwargs: Any
) -> AbstractPos2D:
    """AbstractPos2D -> Cartesian2D -> AbstractPos2D.

    This is the base case for the transformation of 2D vectors.
    """
    return vconvert(target, vconvert(CartesianPos2D, current))


@dispatch.multi(
    (type[CartesianPos2D], CartesianPos2D),
    (type[PolarPos], PolarPos),
)
def vconvert(
    target: type[AbstractPos2D], current: AbstractPos2D, /, **kwargs: Any
) -> AbstractPos2D:
    """Self transform of 2D vectors."""
    return current


@dispatch.multi(
    (type[CartesianVel2D], CartesianVel2D, AbstractPos),
    (type[PolarVel], PolarVel, AbstractPos),
)
def vconvert(
    target: type[AbstractVel2D],
    current: AbstractVel2D,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVel2D:
    """Self transform of 2D Differentials."""
    return current


# =============================================================================
# CartesianPos2D


@dispatch
def vconvert(
    target: type[PolarPos], current: CartesianPos2D, /, **kwargs: Any
) -> PolarPos:
    """CartesianPos2D -> PolarPos.

    The `x` and `y` coordinates are converted to the radial coordinate `r` and
    the angular coordinate `phi`.
    """
    r = xp.hypot(current.x, current.y)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, phi=phi)


# =============================================================================
# PolarPos


@dispatch
def vconvert(
    target: type[CartesianPos2D], current: PolarPos, /, **kwargs: Any
) -> CartesianPos2D:
    """PolarPos -> CartesianPos2D."""
    d = current.r.distance
    x = d * xp.cos(current.phi)
    y = d * xp.sin(current.phi)
    return target(x=x, y=y)


# =============================================================================
# CartesianVel2D


@dispatch
def vconvert(
    target: type[CartesianVel2D], current: CartesianVel2D, /
) -> CartesianVel2D:
    """CartesianVel2D -> CartesianVel2D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.vecs.CartesianVel2D.from_([1, 1], "m/s")
    >>> cx.vconvert(cx.vecs.CartesianVel2D, v) is v
    True

    """
    return current


# =============================================================================
# CartesianAcc2D


@dispatch
def vconvert(
    target: type[CartesianAcc2D], current: CartesianAcc2D, /
) -> CartesianAcc2D:
    """CartesianAcc2D -> CartesianAcc2D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.vecs.CartesianAcc2D.from_([1, 1], "m/s2")
    >>> cx.vconvert(cx.vecs.CartesianAcc2D, a) is a
    True

    """
    return current
