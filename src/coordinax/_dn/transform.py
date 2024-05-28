"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .builtin import (
    CartesianDifferentialND,
    CartesianNDVector,
)

###############################################################################
# 3D


@dispatch
def represent_as(
    current: CartesianNDVector, target: type[CartesianNDVector], /, **kwargs: Any
) -> CartesianNDVector:
    """CartesianNDVector -> CartesianNDVector."""
    return current


@dispatch
def represent_as(
    current: CartesianDifferentialND,
    target: type[CartesianDifferentialND],
    position: CartesianNDVector,
    /,
    **kwargs: Any,
) -> CartesianDifferentialND:
    """CartesianDifferentialND -> CartesianDifferentialND."""
    return current
