"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .cartesian import CartesianAccelerationND, CartesianPositionND, CartesianVelocityND
from .poincare import PoincarePolarVector

###############################################################################
# Cartesian


@dispatch
def represent_as(
    current: CartesianPositionND, target: type[CartesianPositionND], /, **kwargs: Any
) -> CartesianPositionND:
    """CartesianPositionND -> CartesianPositionND.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    >>> q = cx.CartesianPositionND(Quantity([1, 2, 3, 4], "kpc"))

    >>> cx.represent_as(q, cx.CartesianPositionND) is q
    True

    """
    return current


@dispatch
def represent_as(
    current: CartesianVelocityND,
    target: type[CartesianVelocityND],
    position: CartesianPositionND,
    /,
    **kwargs: Any,
) -> CartesianVelocityND:
    """CartesianVelocityND -> CartesianVelocityND.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    >>> x = cx.CartesianPositionND(Quantity([1, 2, 3, 4], "km"))
    >>> v = cx.CartesianVelocityND(Quantity([1, 2, 3, 4], "km/s"))

    >>> cx.represent_as(v, cx.CartesianVelocityND, x) is v
    True

    """
    return current


# =============================================================================
# CartesianVelocityND


@dispatch
def represent_as(
    current: CartesianVelocityND, target: type[CartesianVelocityND], /
) -> CartesianVelocityND:
    """CartesianVelocityND -> CartesianVelocityND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVelocityND.constructor([1, 1, 1], "m/s")
    >>> cx.represent_as(v, cx.CartesianVelocityND) is v
    True

    """
    return current


# =============================================================================
# CartesianAccelerationND


@dispatch
def represent_as(
    current: CartesianAccelerationND, target: type[CartesianAccelerationND], /
) -> CartesianAccelerationND:
    """CartesianAccelerationND -> CartesianAccelerationND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.CartesianAccelerationND.constructor([1, 1, 1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAccelerationND) is a
    True

    """
    return current


###############################################################################
# Poincare


@dispatch
def represent_as(
    current: PoincarePolarVector, target: type[PoincarePolarVector], /, **kwargs: Any
) -> PoincarePolarVector:
    """PoincarePolarVector -> PoincarePolarVector."""
    return current
