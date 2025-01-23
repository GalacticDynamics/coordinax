"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .cartesian import CartesianAccND, CartesianPosND, CartesianVelND
from .poincare import PoincarePolarVector

###############################################################################
# Cartesian


@dispatch
def vconvert(
    target: type[CartesianPosND], current: CartesianPosND, /, **kwargs: Any
) -> CartesianPosND:
    """CartesianPosND -> CartesianPosND.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4], "km"))

    >>> cx.vconvert(cx.vecs.CartesianPosND, q) is q
    True

    """
    return current


@dispatch
def vconvert(
    target: type[CartesianVelND],
    current: CartesianVelND,
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

    >>> cx.vconvert(cx.vecs.CartesianVelND, v, x) is v
    True

    """
    return current


# =============================================================================
# CartesianVelND


@dispatch
def vconvert(
    target: type[CartesianVelND], current: CartesianVelND, /
) -> CartesianVelND:
    """CartesianVelND -> CartesianVelND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.vecs.CartesianVelND.from_([1, 1, 1], "m/s")
    >>> cx.vconvert(cx.vecs.CartesianVelND, v) is v
    True

    """
    return current


# =============================================================================
# CartesianAccND


@dispatch
def vconvert(
    target: type[CartesianAccND], current: CartesianAccND, /
) -> CartesianAccND:
    """CartesianAccND -> CartesianAccND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.vecs.CartesianAccND.from_([1, 1, 1], "m/s2")
    >>> cx.vconvert(cx.vecs.CartesianAccND, a) is a
    True

    """
    return current


###############################################################################
# Poincare


@dispatch
def vconvert(
    target: type[PoincarePolarVector], current: PoincarePolarVector, /, **kwargs: Any
) -> PoincarePolarVector:
    """PoincarePolarVector -> PoincarePolarVector."""
    return current
