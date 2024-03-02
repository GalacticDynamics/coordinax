"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import array_api_jax_compat as xp

from .base import Abstract3DVector, Abstract3DVectorDifferential
from .builtin import (
    Cartesian3DVector,
    CartesianDifferential3D,
    CylindricalDifferential,
    CylindricalVector,
    SphericalDifferential,
    SphericalVector,
)
from coordinax._base import AbstractVector

###############################################################################
# 3D


@dispatch
def represent_as(
    current: Abstract3DVector, target: type[Abstract3DVector], /, **kwargs: Any
) -> Abstract3DVector:
    """Abstract3DVector -> Cartesian3D -> Abstract3DVector."""
    return represent_as(represent_as(current, Cartesian3DVector), target)


@dispatch.multi(
    (Cartesian3DVector, type[Cartesian3DVector]),
    (SphericalVector, type[SphericalVector]),
    (CylindricalVector, type[CylindricalVector]),
)
def represent_as(
    current: Abstract3DVector, target: type[Abstract3DVector], /, **kwargs: Any
) -> Abstract3DVector:
    """Self transform."""
    return current


@dispatch.multi(
    (CartesianDifferential3D, type[CartesianDifferential3D], AbstractVector),
    (SphericalDifferential, type[SphericalDifferential], AbstractVector),
    (CylindricalDifferential, type[CylindricalDifferential], AbstractVector),
)
def represent_as(
    current: Abstract3DVectorDifferential,
    target: type[Abstract3DVectorDifferential],
    position: AbstractVector,
    /,
    **kwargs: Any,
) -> Abstract3DVectorDifferential:
    """Self transform of 3D Differentials."""
    return current


# =============================================================================
# Cartesian3DVector


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[SphericalVector], /, **kwargs: Any
) -> SphericalVector:
    """Cartesian3DVector -> SphericalVector."""
    r = xp.sqrt(current.x**2 + current.y**2 + current.z**2)
    theta = xp.acos(current.z / r)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[CylindricalVector], /, **kwargs: Any
) -> CylindricalVector:
    """Cartesian3DVector -> CylindricalVector."""
    rho = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
    return target(rho=rho, phi=phi, z=current.z)


# =============================================================================
# SphericalVector


@dispatch
def represent_as(
    current: SphericalVector, target: type[Cartesian3DVector], /, **kwargs: Any
) -> Cartesian3DVector:
    """SphericalVector -> Cartesian3DVector."""
    x = current.r * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r * xp.sin(current.theta) * xp.sin(current.phi)
    z = current.r * xp.cos(current.theta)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: SphericalVector, target: type[CylindricalVector], /, **kwargs: Any
) -> CylindricalVector:
    """SphericalVector -> CylindricalVector."""
    rho = xp.abs(current.r * xp.sin(current.theta))
    phi = current.phi
    z = current.r * xp.cos(current.theta)
    return target(rho=rho, phi=phi, z=z)


# =============================================================================
# CylindricalVector


@dispatch
def represent_as(
    current: CylindricalVector, target: type[Cartesian3DVector], /, **kwargs: Any
) -> Cartesian3DVector:
    """CylindricalVector -> Cartesian3DVector."""
    x = current.rho * xp.cos(current.phi)
    y = current.rho * xp.sin(current.phi)
    z = current.z
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: CylindricalVector, target: type[SphericalVector], /, **kwargs: Any
) -> SphericalVector:
    """CylindricalVector -> SphericalVector."""
    r = xp.sqrt(current.rho**2 + current.z**2)
    theta = xp.acos(current.z / r)
    phi = current.phi
    return target(r=r, theta=theta, phi=phi)
