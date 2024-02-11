"""Representation of coordinates in different systems."""

__all__ = ["represent_as"]

from typing import Any
from warnings import warn

import array_api_jax_compat as xp
import astropy.units as u
from jax_quantity import Quantity
from jaxtyping import ArrayLike
from plum import dispatch

from ._base import (  # pylint: disable=cyclic-import
    Abstract1DVector,
    Abstract2DVector,
    Abstract3DVector,
)
from ._builtin import (  # pylint: disable=cyclic-import
    Cartesian1DVector,
    Cartesian2DVector,
    Cartesian3DVector,
    CylindricalVector,
    LnPolarVector,
    Log10PolarVector,
    PolarVector,
    RadialVector,
    SphericalVector,
)
from ._exceptions import IrreversibleDimensionChange

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


@dispatch.multi(
    (RadialVector, type[LnPolarVector]),
    (RadialVector, type[Log10PolarVector]),
)
def represent_as(
    current: Abstract1DVector,
    target: type[Abstract2DVector],
    /,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> Abstract2DVector:
    """Abstract1DVector -> PolarVector -> Abstract2DVector."""
    polar = represent_as(current, PolarVector, phi=phi)
    return represent_as(polar, target)


# =============================================================================
# Cartesian1DVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: Cartesian1DVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """Cartesian1DVector -> RadialVector.

    The `x` coordinate is converted to the radial coordinate `r`.
    """
    return target(r=current.x)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[Cartesian2DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian2DVector:
    """Cartesian1DVector -> Cartesian2DVector.

    The `x` coordinate is converted to the `x` coordinate of the 2D system.
    The `y` coordinate is a keyword argument and defaults to 0.
    """
    return target(x=current.x, y=y)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[PolarVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> PolarVector:
    """Cartesian1DVector -> PolarVector.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.
    """
    return target(r=current.x, phi=phi)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[LnPolarVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> LnPolarVector:
    """Cartesian1DVector -> LnPolarVector.

    The `x` coordinate is converted to the radial coordinate `lnr`.
    The `phi` coordinate is a keyword argument and defaults to 0.
    """
    return target(lnr=xp.log(current.x), phi=phi)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[Log10PolarVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> Log10PolarVector:
    """Cartesian1DVector -> Log10PolarVector.

    The `x` coordinate is converted to the radial coordinate `log10r`.
    The `phi` coordinate is a keyword argument and defaults to 0.
    """
    return target(log10r=xp.log10(current.x), phi=phi)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[Cartesian3DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian3DVector:
    """Cartesian1DVector -> Cartesian3DVector.

    The `x` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.
    """
    return target(x=current.x, y=y, z=z)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[SphericalVector],
    /,
    *,
    theta: Quantity = Quantity(0.0, u.radian),
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> SphericalVector:
    """Cartesian1DVector -> SphericalVector.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.
    """
    return target(r=current.x, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: Cartesian1DVector,
    target: type[CylindricalVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> CylindricalVector:
    """Cartesian1DVector -> CylindricalVector.

    The `x` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.
    """
    return target(rho=current.x, phi=phi, z=z)


# =============================================================================
# RadialVector

# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: RadialVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """RadialVector -> Cartesian1DVector.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.
    """
    return target(x=current.r)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: RadialVector,
    target: type[Cartesian2DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian2DVector:
    """RadialVector -> Cartesian2DVector.

    The `r` coordinate is converted to the cartesian coordinate `x`.
    The `y` coordinate is a keyword argument and defaults to 0.
    """
    return target(x=current.r, y=y)


@dispatch
def represent_as(
    current: RadialVector,
    target: type[PolarVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> PolarVector:
    """RadialVector -> PolarVector.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.
    """
    return target(r=current.r, phi=phi)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: RadialVector,
    target: type[Cartesian3DVector],
    /,
    *,
    y: Quantity = Quantity(0.0, u.m),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian3DVector:
    """RadialVector -> Cartesian3DVector.

    The `r` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.
    """
    return target(x=current.r, y=y, z=z)


@dispatch
def represent_as(
    current: RadialVector,
    target: type[SphericalVector],
    /,
    *,
    theta: ArrayLike = 0.0,
    phi: Quantity = Quantity(0.0, u.radian),
    **kwargs: Any,
) -> SphericalVector:
    """RadialVector -> SphericalVector.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.
    """
    return target(r=current.r, theta=theta, phi=phi)


@dispatch
def represent_as(
    current: RadialVector,
    target: type[CylindricalVector],
    /,
    *,
    phi: Quantity = Quantity(0.0, u.radian),
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> CylindricalVector:
    """RadialVector -> CylindricalVector.

    The `r` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.
    """
    return target(rho=current.r, phi=phi, z=z)


###############################################################################
# 2D


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
    (LnPolarVector, type[LnPolarVector]),
    (Log10PolarVector, type[Log10PolarVector]),
)
def represent_as(
    current: Abstract2DVector, target: type[Abstract2DVector], /, **kwargs: Any
) -> Abstract2DVector:
    """Self transform of 2D vectors."""
    return current


@dispatch.multi(
    (Cartesian2DVector, type[LnPolarVector]),
    (Cartesian2DVector, type[Log10PolarVector]),
    (LnPolarVector, type[Cartesian2DVector]),
    (Log10PolarVector, type[Cartesian2DVector]),
)
def represent_as(
    current: Abstract2DVector, target: type[Abstract2DVector], /, **kwargs: Any
) -> Abstract2DVector:
    """Abstract2DVector -> PolarVector -> Abstract2DVector."""
    polar = represent_as(current, PolarVector)
    return represent_as(polar, target)


@dispatch.multi(
    (Cartesian2DVector, type[SphericalVector]),
    (Cartesian2DVector, type[CylindricalVector]),
)
def represent_as(
    current: Abstract2DVector,
    target: type[Abstract3DVector],
    /,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Abstract3DVector:
    """Abstract2DVector -> Cartesian2D -> Cartesian3D -> Abstract3DVector.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.
    """
    cart2 = represent_as(current, Cartesian2DVector)
    cart3 = represent_as(cart2, Cartesian3DVector, z=z)
    return represent_as(cart3, target)


@dispatch.multi(
    (PolarVector, type[Cartesian3DVector]),
    (PolarVector, type[SphericalVector]),
    (LnPolarVector, type[Cartesian3DVector]),
    (LnPolarVector, type[CylindricalVector]),
    (LnPolarVector, type[SphericalVector]),
    (Log10PolarVector, type[Cartesian3DVector]),
    (Log10PolarVector, type[CylindricalVector]),
    (Log10PolarVector, type[SphericalVector]),
)
def represent_as(
    current: Abstract2DVector,
    target: type[Abstract3DVector],
    /,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Abstract3DVector:
    """Abstract2DVector -> PolarVector -> Cylindrical -> Abstract3DVector.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.
    """
    polar = represent_as(current, PolarVector)
    cyl = represent_as(polar, CylindricalVector, z=z)
    return represent_as(cyl, target)


# =============================================================================
# Cartesian2DVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: Cartesian2DVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """Cartesian2DVector -> Cartesian1DVector.

    The `y` coordinate is dropped.
    """
    warn(
        "The y coordinate is being dropped.", IrreversibleDimensionChange, stacklevel=2
    )
    return target(x=current.x)


@dispatch
def represent_as(
    current: Cartesian2DVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """Cartesian2DVector -> RadialVector.

    The `x` and `y` coordinates are converted to the radial coordinate `r`.
    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=xp.sqrt(current.x**2 + current.y**2))


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


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: Cartesian2DVector,
    target: type[Cartesian3DVector],
    /,
    *,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> Cartesian3DVector:
    """Cartesian2DVector -> Cartesian3DVector.

    The `x` and `y` coordinates are converted to the `x` and `y` coordinates of
    the 3D system.  The `z` coordinate is a keyword argument and defaults to 0.
    """
    return target(x=current.x, y=current.y, z=z)


# =============================================================================
# PolarVector

# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: PolarVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """PolarVector -> Cartesian1DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * xp.cos(current.phi))


@dispatch
def represent_as(
    current: PolarVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """PolarVector -> RadialVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


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


@dispatch
def represent_as(
    current: PolarVector, target: type[LnPolarVector], /, **kwargs: Any
) -> LnPolarVector:
    """PolarVector -> LnPolarVector."""
    return target(lnr=xp.log(current.r), phi=current.phi)


@dispatch
def represent_as(
    current: PolarVector, target: type[Log10PolarVector], /, **kwargs: Any
) -> Log10PolarVector:
    """PolarVector -> Log10PolarVector."""
    return target(log10r=xp.log10(current.r), phi=current.phi)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: PolarVector,
    target: type[CylindricalVector],
    /,
    *,
    z: Quantity = Quantity(0.0, u.m),
    **kwargs: Any,
) -> CylindricalVector:
    """PolarVector -> CylindricalVector."""
    return target(rho=current.r, phi=current.phi, z=z)


# =============================================================================
# LnPolarVector

# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: LnPolarVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """LnPolarVector -> Cartesian1DVector."""
    polar = represent_as(current, PolarVector)
    return represent_as(polar, target)


@dispatch
def represent_as(
    current: LnPolarVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """LnPolarVector -> RadialVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=xp.exp(current.lnr))


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: LnPolarVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """LnPolarVector -> PolarVector."""
    return target(r=xp.exp(current.lnr), phi=current.phi)


@dispatch
def represent_as(
    current: LnPolarVector, target: type[Log10PolarVector], /, **kwargs: Any
) -> Log10PolarVector:
    """LnPolarVector -> Log10PolarVector."""
    return target(log10r=current.lnr / xp.log(10), phi=current.phi)


# =============================================================================
# Log10PolarVector

# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: Log10PolarVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """Log10PolarVector -> Cartesian1DVector."""
    # warn("irreversible dimension change", IrreversibleDimensionChange,
    # stacklevel=2)
    polar = represent_as(current, PolarVector)
    return represent_as(polar, target)


@dispatch
def represent_as(
    current: Log10PolarVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """Log10PolarVector -> RadialVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=xp.pow(10, current.log10r))


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: Log10PolarVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """Log10PolarVector -> PolarVector."""
    return target(r=xp.pow(10, current.log10r), phi=current.phi)


@dispatch
def represent_as(
    current: Log10PolarVector, target: type[LnPolarVector], /, **kwargs: Any
) -> LnPolarVector:
    """Log10PolarVector -> LnPolarVector."""
    return target(lnr=current.log10r * xp.log(10), phi=current.phi)


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
    (CylindricalVector, type[LnPolarVector]),
    (CylindricalVector, type[Log10PolarVector]),
    (SphericalVector, type[LnPolarVector]),
    (SphericalVector, type[Log10PolarVector]),
)
def represent_as(
    current: Abstract3DVector, target: type[Abstract2DVector], **kwargs: Any
) -> Abstract2DVector:
    """Abstract3DVector -> Cylindrical -> PolarVector -> Abstract2DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    cyl = represent_as(current, CylindricalVector)
    polar = represent_as(cyl, PolarVector)
    return represent_as(polar, target)


# =============================================================================
# Cartesian3DVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """Cartesian3DVector -> Cartesian1DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x)


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """Cartesian3DVector -> RadialVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=xp.sqrt(current.x**2 + current.y**2 + current.z**2))


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """Cartesian3DVector -> Cartesian2DVector."""
    warn(
        "The z coordinate is being dropped.", IrreversibleDimensionChange, stacklevel=2
    )
    return target(x=current.x, y=current.y)


@dispatch.multi(
    (Cartesian3DVector, type[PolarVector]),
    (Cartesian3DVector, type[LnPolarVector]),
    (Cartesian3DVector, type[Log10PolarVector]),
)
def represent_as(
    current: Cartesian3DVector, target: type[Abstract2DVector], /, **kwargs: Any
) -> Abstract2DVector:
    """Cartesian3DVector -> Cartesian2D -> Abstract2DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    cart2 = represent_as(current, Cartesian2DVector)
    return represent_as(cart2, target)


# -----------------------------------------------
# 3D


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


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: SphericalVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """SphericalVector -> Cartesian1DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.r * xp.sin(current.theta) * xp.cos(current.phi))


@dispatch
def represent_as(
    current: SphericalVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """SphericalVector -> RadialVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: SphericalVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """SphericalVector -> Cartesian2DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r * xp.sin(current.theta) * xp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: SphericalVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """SphericalVector -> PolarVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.r * xp.sin(current.theta), phi=current.phi)


# -----------------------------------------------
# 3D


@dispatch
def represent_as(
    current: SphericalVector, target: type[Cartesian3DVector], /, **kwargs: Any
) -> Cartesian3DVector:
    """SphericalVector -> Cartesian3DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.r * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r * xp.sin(current.theta) * xp.sin(current.phi)
    z = current.r * xp.cos(current.theta)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: SphericalVector, target: type[CylindricalVector], /, **kwargs: Any
) -> CylindricalVector:
    """SphericalVector -> CylindricalVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    rho = current.r * xp.sin(current.theta)
    phi = current.phi
    z = current.r * xp.cos(current.theta)
    return target(rho=rho, phi=phi, z=z)


# =============================================================================
# CylindricalVector


# -----------------------------------------------
# 1D


@dispatch
def represent_as(
    current: CylindricalVector, target: type[Cartesian1DVector], /, **kwargs: Any
) -> Cartesian1DVector:
    """CylindricalVector -> Cartesian1DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.rho * xp.cos(current.phi))


@dispatch
def represent_as(
    current: CylindricalVector, target: type[RadialVector], /, **kwargs: Any
) -> RadialVector:
    """CylindricalVector -> RadialVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho)


# -----------------------------------------------
# 2D


@dispatch
def represent_as(
    current: CylindricalVector, target: type[Cartesian2DVector], /, **kwargs: Any
) -> Cartesian2DVector:
    """CylindricalVector -> Cartesian2DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = current.rho * xp.cos(current.phi)
    y = current.rho * xp.sin(current.phi)
    return target(x=x, y=y)


@dispatch
def represent_as(
    current: CylindricalVector, target: type[PolarVector], /, **kwargs: Any
) -> PolarVector:
    """CylindricalVector -> PolarVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(r=current.rho, phi=current.phi)


# -----------------------------------------------
# 3D


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
