"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

from .base import Abstract1DVector
from .builtin import Cartesian1DVector, RadialVector

###############################################################################
# 1D


@dispatch
def represent_as(
    current: Abstract1DVector, target: type[Abstract1DVector], /, **kwargs: Any
) -> Abstract1DVector:
    """Abstract1DVector -> Cartesian1D -> Abstract1DVector.

    This is the base case for the transformation of 1D vectors.
    """
    cart1d = represent_as(current, Cartesian1DVector)
    return represent_as(cart1d, target)


@dispatch.multi(
    (Cartesian1DVector, type[Cartesian1DVector]), (RadialVector, type[RadialVector])
)
def represent_as(
    current: Abstract1DVector, target: type[Abstract1DVector], /, **kwargs: Any
) -> Abstract1DVector:
    """Self transform of 1D vectors."""
    return current


# =============================================================================
# Cartesian1DVector


@dispatch
def represent_as(
    current: Cartesian1DVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """Cartesian1DVector -> RadialVector.

    The `x` coordinate is converted to the radial coordinate `r`.
    """
    return target(r=current.x)


# =============================================================================
# RadialVector


@dispatch
def represent_as(
    current: RadialVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """RadialVector -> Cartesian1DVector.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.
    """
    return target(x=current.r)
