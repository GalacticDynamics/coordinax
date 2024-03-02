"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import array_api_jax_compat as xp

from .base import Abstract2DVector, Abstract2DVectorDifferential
from .builtin import (
    Cartesian2DVector,
    CartesianDifferential2D,
    PolarDifferential,
    PolarVector,
)
from coordinax._base import AbstractVector


@dispatch
def represent_as(
    current: Abstract2DVector, target: type[Abstract2DVector], /, **kwargs: Any
) -> Abstract2DVector:
    """Abstract2DVector -> Cartesian2D -> Abstract2DVector.

    This is the base case for the transformation of 2D vectors.
    """
    return represent_as(represent_as(current, Cartesian2DVector), target)


@dispatch.multi(
    (Cartesian2DVector, type[Cartesian2DVector]),
    (PolarVector, type[PolarVector]),
    # (LnPolarVector, type[LnPolarVector]),
    # (Log10PolarVector, type[Log10PolarVector]),
)
def represent_as(
    current: Abstract2DVector, target: type[Abstract2DVector], /, **kwargs: Any
) -> Abstract2DVector:
    """Self transform of 2D vectors."""
    return current


@dispatch.multi(
    (CartesianDifferential2D, type[CartesianDifferential2D], AbstractVector),
    (PolarDifferential, type[PolarDifferential], AbstractVector),
)
def represent_as(
    current: Abstract2DVectorDifferential,
    target: type[Abstract2DVectorDifferential],
    position: AbstractVector,
    /,
    **kwargs: Any,
) -> Abstract2DVectorDifferential:
    """Self transform of 2D Differentials."""
    return current


# @dispatch.multi(
#     (Cartesian2DVector, type[LnPolarVector]),
#     (Cartesian2DVector, type[Log10PolarVector]),
#     (LnPolarVector, type[Cartesian2DVector]),
#     (Log10PolarVector, type[Cartesian2DVector]),
# )
# def represent_as(
#     current: Abstract2DVector, target: type[Abstract2DVector], /, **kwargs: Any
# ) -> Abstract2DVector:
#     """Abstract2DVector -> PolarVector -> Abstract2DVector."""
#     polar = represent_as(current, PolarVector)
#     return represent_as(polar, target)


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
    x = current.r * xp.cos(current.phi)
    y = current.r * xp.sin(current.phi)
    return target(x=x, y=y)


# @dispatch
# def represent_as(
#     current: PolarVector, target: type[LnPolarVector], /, **kwargs: Any
# ) -> LnPolarVector:
#     """PolarVector -> LnPolarVector."""
#     return target(lnr=xp.log(current.r), phi=current.phi)


# @dispatch
# def represent_as(
#     current: PolarVector, target: type[Log10PolarVector], /, **kwargs: Any
# ) -> Log10PolarVector:
#     """PolarVector -> Log10PolarVector."""
#     return target(log10r=xp.log10(current.r), phi=current.phi)


# # =============================================================================
# # LnPolarVector

# # -----------------------------------------------
# # 2D


# @dispatch
# def represent_as(
#     current: LnPolarVector, target: type[PolarVector], /, **kwargs: Any
# ) -> PolarVector:
#     """LnPolarVector -> PolarVector."""
#     return target(r=xp.exp(current.lnr), phi=current.phi)


# @dispatch
# def represent_as(
#     current: LnPolarVector, target: type[Log10PolarVector], /, **kwargs: Any
# ) -> Log10PolarVector:
#     """LnPolarVector -> Log10PolarVector."""
#     return target(log10r=current.lnr / xp.log(10), phi=current.phi)


# # =============================================================================
# # Log10PolarVector

# # -----------------------------------------------
# # 2D


# @dispatch
# def represent_as(
#     current: Log10PolarVector, target: type[PolarVector], /, **kwargs: Any
# ) -> PolarVector:
#     """Log10PolarVector -> PolarVector."""
#     return target(r=xp.pow(10, current.log10r), phi=current.phi)


# @dispatch
# def represent_as(
#     current: Log10PolarVector, target: type[LnPolarVector], /, **kwargs: Any
# ) -> LnPolarVector:
#     """Log10PolarVector -> LnPolarVector."""
#     return target(lnr=current.log10r * xp.log(10), phi=current.phi)
