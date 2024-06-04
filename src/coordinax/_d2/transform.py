"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.array_api as xp

from .base import AbstractPosition2D, AbstractVelocity2D
from .cartesian import CartesianPosition2D, CartesianVelocity2D
from .polar import PolarPosition, PolarVelocity
from coordinax._base_pos import AbstractPosition


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

# -----------------------------------------------
# 2D


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

# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: PolarPosition, target: type[CartesianPosition2D], /, **kwargs: Any
) -> CartesianPosition2D:
    """PolarPosition -> CartesianPosition2D."""
    x = current.r.distance * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.phi)
    return target(x=x, y=y)
