"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .builtin import CartesianPositionND, CartesianVelocityND

###############################################################################
# 3D


@dispatch
def represent_as(
    current: CartesianPositionND, target: type[CartesianPositionND], /, **kwargs: Any
) -> CartesianPositionND:
    """CartesianPositionND -> CartesianPositionND."""
    return current


@dispatch
def represent_as(
    current: CartesianVelocityND,
    target: type[CartesianVelocityND],
    position: CartesianPositionND,
    /,
    **kwargs: Any,
) -> CartesianVelocityND:
    """CartesianVelocityND -> CartesianVelocityND."""
    return current
