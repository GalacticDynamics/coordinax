"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .base import AbstractPosition1D, AbstractVelocity1D
from .builtin import (
    CartesianPosition1D,
    CartesianVelocity1D,
    RadialPosition,
    RadialVelocity,
)
from coordinax._base_pos import AbstractPosition

###############################################################################
# 1D


@dispatch(precedence=1)
def represent_as(
    current: CartesianPosition1D, target: type[CartesianPosition1D], /, **kwargs: Any
) -> CartesianPosition1D:
    """Self transform of 1D vectors."""
    return current


@dispatch(precedence=1)
def represent_as(
    current: RadialPosition, target: type[RadialPosition], /, **kwargs: Any
) -> RadialPosition:
    """Self transform of 1D vectors."""
    return current


@dispatch
def represent_as(
    current: AbstractPosition1D, target: type[AbstractPosition1D], /, **kwargs: Any
) -> AbstractPosition1D:
    """AbstractPosition1D -> Cartesian1D -> AbstractPosition1D.

    This is the base case for the transformation of 1D vectors.
    """
    cart1d = represent_as(current, CartesianPosition1D)
    return represent_as(cart1d, target)


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
    """Self transform of 1D Differentials."""
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
