"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .cartesian import CartesianAccND, CartesianPosND, CartesianVelND
from .poincare import PoincarePolarVector

###############################################################################
# Cartesian


@dispatch
def represent_as(
    current: CartesianPosND, target: type[CartesianPosND], /, **kwargs: Any
) -> CartesianPosND:
    """CartesianPosND -> CartesianPosND.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4], "km"))

    >>> cx.represent_as(q, cx.vecs.CartesianPosND) is q
    True

    """
    return current


@dispatch
def represent_as(
    current: CartesianVelND,
    target: type[CartesianVelND],
    position: CartesianPosND,
    /,
    **kwargs: Any,
) -> CartesianVelND:
    """CartesianVelND -> CartesianVelND.

    Examples
    --------
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPosND.from_([1, 2, 3, 4], "km")
    >>> v = cx.vecs.CartesianVelND.from_([1, 2, 3, 4], "km/s")

    >>> cx.represent_as(v, cx.vecs.CartesianVelND, x) is v
    True

    """
    return current


# =============================================================================
# CartesianVelND


@dispatch
def represent_as(
    current: CartesianVelND, target: type[CartesianVelND], /
) -> CartesianVelND:
    """CartesianVelND -> CartesianVelND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.vecs.CartesianVelND.from_([1, 1, 1], "m/s")
    >>> cx.represent_as(v, cx.vecs.CartesianVelND) is v
    True

    """
    return current


# =============================================================================
# CartesianAccND


@dispatch
def represent_as(
    current: CartesianAccND, target: type[CartesianAccND], /
) -> CartesianAccND:
    """CartesianAccND -> CartesianAccND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.vecs.CartesianAccND.from_([1, 1, 1], "m/s2")
    >>> cx.represent_as(a, cx.vecs.CartesianAccND) is a
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
