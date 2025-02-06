"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.numpy as xp

from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from .cartesian import CartesianAcc2D, CartesianPos2D, CartesianVel2D
from .polar import PolarAcc, PolarPos, PolarVel
from .spherical import TwoSphereAcc, TwoSpherePos, TwoSphereVel
from coordinax._src.vectors.base_pos import AbstractPos


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


###############################################################################
# Corresponding Cartesian classes


@dispatch
def cartesian_vector_type(
    obj: type[AbstractPos2D] | AbstractPos2D, /
) -> type[CartesianPos2D]:
    """AbstractPos2D -> CartesianPos2D."""
    return CartesianPos2D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractVel2D] | AbstractVel2D, /
) -> type[CartesianVel2D]:
    """AbstractVel2D -> CartesianVel2D."""
    return CartesianVel2D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractAcc2D] | AbstractAcc2D, /
) -> type[CartesianAcc2D]:
    """AbstractPos -> CartesianAcc2D."""
    return CartesianAcc2D


###############################################################################
# Corresponding time derivative classes

# -----------------------------------------------
# Position -> Velocity


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianPos2D] | CartesianPos2D, /
) -> type[CartesianVel2D]:
    """Return the corresponding time derivative class."""
    return CartesianVel2D


@dispatch
def time_derivative_vector_type(obj: type[PolarPos] | PolarPos, /) -> type[PolarVel]:
    """Return the corresponding time derivative class."""
    return PolarVel


@dispatch
def time_derivative_vector_type(
    obj: type[TwoSpherePos] | TwoSpherePos, /
) -> type[TwoSphereVel]:
    """Return the corresponding time derivative class."""
    return TwoSphereVel


# -----------------------------------------------
# Velocity -> Position


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianVel2D] | CartesianVel2D, /
) -> type[CartesianPos2D]:
    """Return the corresponding time antiderivative class."""
    return CartesianPos2D


@dispatch
def time_antiderivative_vector_type(
    obj: type[PolarVel] | PolarVel, /
) -> type[PolarPos]:
    """Return the corresponding time antiderivative class."""
    return PolarPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[TwoSphereVel] | TwoSphereVel, /
) -> type[TwoSpherePos]:
    """Return the corresponding time antiderivative class."""
    return TwoSpherePos


# -----------------------------------------------
# Velocity -> Acceleration


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianVel2D] | CartesianVel2D, /
) -> type[CartesianAcc2D]:
    """Return the corresponding time derivative class."""
    return CartesianAcc2D


@dispatch
def time_derivative_vector_type(obj: type[PolarVel] | PolarVel, /) -> type[PolarAcc]:
    """Return the corresponding time derivative class."""
    return PolarAcc


@dispatch
def time_derivative_vector_type(
    obj: type[TwoSphereVel] | TwoSphereVel, /
) -> type[TwoSphereAcc]:
    """Return the corresponding time derivative class."""
    return TwoSphereAcc


# -----------------------------------------------
# Acceleration -> Velocity


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianAcc2D] | CartesianAcc2D, /
) -> type[CartesianVel2D]:
    """Return the corresponding time antiderivative class."""
    return CartesianVel2D


@dispatch
def time_antiderivative_vector_type(
    obj: type[PolarAcc] | PolarAcc, /
) -> type[PolarVel]:
    """Return the corresponding time antiderivative class."""
    return PolarVel


@dispatch
def time_antiderivative_vector_type(
    obj: type[TwoSphereAcc] | TwoSphereAcc, /
) -> type[TwoSphereVel]:
    """Return the corresponding time antiderivative class."""
    return TwoSphereVel
