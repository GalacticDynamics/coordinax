"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .builtin import CartesianPositionND, CartesianVelocityND
from .poincare import PoincarePolarVector

###############################################################################
# N-Dimensional


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


###############################################################################
# Poincare


@dispatch
def represent_as(
    current: PoincarePolarVector, target: type[PoincarePolarVector], /, **kwargs: Any
) -> PoincarePolarVector:
    """PoincarePolarVector -> PoincarePolarVector."""
    return current
