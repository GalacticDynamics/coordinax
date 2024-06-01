"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.array_api as xp

from .base import AbstractPosition2D, AbstractVelocity2D
from .builtin import (
    Cartesian2DVector,
    CartesianDifferential2D,
    PolarDifferential,
    PolarVector,
)
from coordinax._base_pos import AbstractPosition


@dispatch
def represent_as(
    current: AbstractPosition2D, target: type[AbstractPosition2D], /, **kwargs: Any
) -> AbstractPosition2D:
    """AbstractPosition2D -> Cartesian2D -> AbstractPosition2D.

    This is the base case for the transformation of 2D vectors.
    """
    return represent_as(represent_as(current, Cartesian2DVector), target)


@dispatch.multi(
    (Cartesian2DVector, type[Cartesian2DVector]),
    (PolarVector, type[PolarVector]),
)
def represent_as(
    current: AbstractPosition2D, target: type[AbstractPosition2D], /, **kwargs: Any
) -> AbstractPosition2D:
    """Self transform of 2D vectors."""
    return current


@dispatch.multi(
    (CartesianDifferential2D, type[CartesianDifferential2D], AbstractPosition),
    (PolarDifferential, type[PolarDifferential], AbstractPosition),
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
# Cartesian2DVector

# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: Cartesian2DVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """Cartesian2DVector -> PolarVector.

    The `x` and `y` coordinates are converted to the radial coordinate `r` and
    the angular coordinate `phi`.
    """
    r = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, phi=phi)


# =============================================================================
# PolarVector

# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: PolarVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """PolarVector -> Cartesian2DVector."""
    x = current.r.distance * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.phi)
    return target(x=x, y=y)
