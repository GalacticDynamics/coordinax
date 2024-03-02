"""Representation of coordinates in different systems."""

__all__ = ["represent_as"]

from math import prod
from typing import Any
from warnings import warn

import astropy.units as u
import jax
from plum import dispatch

import array_api_jax_compat as xp
from jax_quantity import Quantity

from ._base import AbstractVector, AbstractVectorDifferential
from ._d1.base import Abstract1DVectorDifferential
from ._d1.builtin import Cartesian1DVector, RadialVector
from ._d2.base import Abstract2DVector, Abstract2DVectorDifferential
from ._d2.builtin import Cartesian2DVector, PolarVector
from ._d3.base import Abstract3DVector, Abstract3DVectorDifferential
from ._d3.builtin import Cartesian3DVector, CylindricalVector, SphericalVector
from ._exceptions import IrreversibleDimensionChange
from ._utils import dataclass_items


# TODO: implement for cross-representations
@dispatch.multi(  # type: ignore[misc]
    # N-D -> N-D
    (
        Abstract1DVectorDifferential,
        type[Abstract1DVectorDifferential],  # type: ignore[misc]
        AbstractVector,
    ),
    (
        Abstract2DVectorDifferential,
        type[Abstract2DVectorDifferential],  # type: ignore[misc]
        AbstractVector,
    ),
    (
        Abstract3DVectorDifferential,
        type[Abstract3DVectorDifferential],  # type: ignore[misc]
        AbstractVector,
    ),
)
def represent_as(
    current: AbstractVectorDifferential,
    target: type[AbstractVectorDifferential],
    position: AbstractVector,
    /,
    **kwargs: Any,
) -> AbstractVectorDifferential:
    """Abstract3DVectorDifferential -> Cartesian -> Abstract3DVectorDifferential.

    This is the base case for the transformation of 1D vector differentials.
    """
    # TODO: not require the shape munging / support more shapes
    shape = current.shape
    flat_shape = prod(shape)
    position = position.reshape(flat_shape)  # flattened

    # Start by transforming the position to the type required by the
    # differential to construct the Jacobian.
    current_position = represent_as(position, current.integral_cls, **kwargs)

    # Takes the Jacobian through the representation transformation function.  This
    # returns a representation of the target type, where the value of each field the
    # corresponding row of the Jacobian. The value of the field is a Quantity with
    # the correct numerator unit (of the Jacobian row). The value is a Vector of the
    # original type, with fields that are the columns of that row, but with only the
    # denomicator's units.
    jac_nested_vecs = jac_rep_as(current_position, target.integral_cls)

    # This changes the Jacobian to be a dictionary of each row, with the value
    # being that row's column as a dictionary, now with the correct units for
    # each element:  {row_i: {col_j: Quantity(value, row.unit / column.unit)}}
    jac_rows = {
        f"d_{k}": {
            kk: Quantity(vv.value, unit=v.unit / vv.unit)
            for kk, vv in dataclass_items(v.value)
        }
        for k, v in dataclass_items(jac_nested_vecs)
    }

    # Now we can use the Jacobian to transform the differential.
    flat_current = current.reshape(flat_shape)
    return target(
        **{  # Each field is the dot product of the row of the J and the diff column.
            k: xp.sum(  # Doing the dot product.
                xp.stack(
                    tuple(
                        j_c * getattr(flat_current, f"d_{kk}")
                        for kk, j_c in j_r.items()
                    )
                ),
                axis=0,
            )
            for k, j_r in jac_rows.items()
        }
    ).reshape(shape)


# TODO: situate this better to show how represent_as is used
jac_rep_as = jax.jit(
    jax.vmap(jax.jacfwd(represent_as), in_axes=(0, None)), static_argnums=(1,)
)


###############################################################################
# 1D


# @dispatch.multi(
#     (RadialVector, type[LnPolarVector]),
#     (RadialVector, type[Log10PolarVector]),
# )
# def represent_as(
#     current: Abstract1DVector,
#     target: type[Abstract2DVector],
#     /,
#     phi: Quantity = Quantity(0.0, u.radian),
#     **kwargs: Any,
# ) -> Abstract2DVector:
#     """Abstract1DVector -> PolarVector -> Abstract2DVector."""
#     polar = represent_as(current, PolarVector, phi=phi)
#     return represent_as(polar, target)


# =============================================================================
# Cartesian1DVector


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


# @dispatch
# def represent_as(
#     current: Cartesian1DVector,
#     target: type[LnPolarVector],
#     /,
#     *,
#     phi: Quantity = Quantity(0.0, u.radian),
#     **kwargs: Any,
# ) -> LnPolarVector:
#     """Cartesian1DVector -> LnPolarVector.

#     The `x` coordinate is converted to the radial coordinate `lnr`.
#     The `phi` coordinate is a keyword argument and defaults to 0.
#     """
#     return target(lnr=xp.log(current.x), phi=phi)


# @dispatch
# def represent_as(
#     current: Cartesian1DVector,
#     target: type[Log10PolarVector],
#     /,
#     *,
#     phi: Quantity = Quantity(0.0, u.radian),
#     **kwargs: Any,
# ) -> Log10PolarVector:
#     """Cartesian1DVector -> Log10PolarVector.

#     The `x` coordinate is converted to the radial coordinate `log10r`.
#     The `phi` coordinate is a keyword argument and defaults to 0.
#     """
#     return target(log10r=xp.log10(current.x), phi=phi)


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
    theta: Quantity = Quantity(0.0, u.radian),
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
    # (LnPolarVector, type[Cartesian3DVector]),
    # (LnPolarVector, type[CylindricalVector]),
    # (LnPolarVector, type[SphericalVector]),
    # (Log10PolarVector, type[Cartesian3DVector]),
    # (Log10PolarVector, type[CylindricalVector]),
    # (Log10PolarVector, type[SphericalVector]),
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
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
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
# 3D


@dispatch
def represent_as(
    current: PolarVector,
    target: type[SphericalVector],
    /,
    theta: Quantity["angle"] = Quantity(0.0, u.radian),  # type: ignore[name-defined]
    **kwargs: Any,
) -> SphericalVector:
    """PolarVector -> SphericalVector."""
    return target(r=current.r, theta=theta, phi=current.phi)


@dispatch
def represent_as(
    current: PolarVector,
    target: type[CylindricalVector],
    /,
    *,
    z: Quantity["length"] = Quantity(0.0, u.m),  # type: ignore[name-defined]
    **kwargs: Any,
) -> CylindricalVector:
    """PolarVector -> CylindricalVector."""
    return target(rho=current.r, phi=current.phi, z=z)


# # =============================================================================
# # LnPolarVector

# # -----------------------------------------------
# # 1D


# @dispatch
# def represent_as(
#     current: LnPolarVector, target: type[Cartesian1DVector], /, **kwargs: Any
# ) -> Cartesian1DVector:
#     """LnPolarVector -> Cartesian1DVector."""
#     polar = represent_as(current, PolarVector)
#     return represent_as(polar, target)


# @dispatch
# def represent_as(
#     current: LnPolarVector, target: type[RadialVector], /, **kwargs: Any
# ) -> RadialVector:
#     """LnPolarVector -> RadialVector."""
#     warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
#     return target(r=xp.exp(current.lnr))


# # =============================================================================
# # Log10PolarVector

# # -----------------------------------------------
# # 1D


# @dispatch
# def represent_as(
#     current: Log10PolarVector, target: type[Cartesian1DVector], /, **kwargs: Any
# ) -> Cartesian1DVector:
#     """Log10PolarVector -> Cartesian1DVector."""
#     # warn("irreversible dimension change", IrreversibleDimensionChange,
#     # stacklevel=2)
#     polar = represent_as(current, PolarVector)
#     return represent_as(polar, target)


# @dispatch
# def represent_as(
#     current: Log10PolarVector, target: type[RadialVector], /, **kwargs: Any
# ) -> RadialVector:
#     """Log10PolarVector -> RadialVector."""
#     warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
#     return target(r=xp.pow(10, current.log10r))


###############################################################################
# 3D


# @dispatch.multi(
#     (CylindricalVector, type[LnPolarVector]),
#     (CylindricalVector, type[Log10PolarVector]),
#     (SphericalVector, type[LnPolarVector]),
#     (SphericalVector, type[Log10PolarVector]),
# )
# def represent_as(
#     current: Abstract3DVector, target: type[Abstract2DVector], **kwargs: Any
# ) -> Abstract2DVector:
#     """Abstract3DVector -> Cylindrical -> PolarVector -> Abstract2DVector."""
#     warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
#     cyl = represent_as(current, CylindricalVector)
#     polar = represent_as(cyl, PolarVector)
#     return represent_as(polar, target)


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
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return target(x=current.x, y=current.y)


@dispatch.multi(
    (Cartesian3DVector, type[PolarVector]),
    # (Cartesian3DVector, type[LnPolarVector]),
    # (Cartesian3DVector, type[Log10PolarVector]),
)
def represent_as(
    current: Cartesian3DVector, target: type[Abstract2DVector], /, **kwargs: Any
) -> Abstract2DVector:
    """Cartesian3DVector -> Cartesian2D -> Abstract2DVector."""
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    cart2 = represent_as(current, Cartesian2DVector)
    return represent_as(cart2, target)


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
