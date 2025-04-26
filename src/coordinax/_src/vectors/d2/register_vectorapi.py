"""Representation of coordinates in different systems."""

__all__: list[str] = []

import functools as ft

import jax
from plum import dispatch

import quaxed.numpy as jnp

import coordinax._src.vectors.custom_types as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from .cartesian import CartesianAcc2D, CartesianPos2D, CartesianVel2D
from .polar import PolarAcc, PolarPos, PolarVel
from .spherical import TwoSphereAcc, TwoSpherePos, TwoSphereVel
from coordinax._src.vectors.private_api import combine_aux, wrap_vconvert_impl_params

###############################################################################
# Vector Transformation


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert(
    to_vector: type[AbstractPos2D],
    from_vector: type[AbstractPos2D],
    params: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """AbstractPos -> CartesianPos1D -> AbstractPos."""
    params, aux = vconvert(
        CartesianPos2D, from_vector, params, in_aux=in_aux, out_aux=None, units=units
    )
    params, aux = vconvert(
        to_vector, CartesianPos2D, params, in_aux=aux, out_aux=out_aux, units=units
    )
    return params, aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[PolarPos],
    from_vector: type[CartesianPos2D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.OptAuxDict]:
    """CartesianPos2D -> PolarPos.

    The `x` and `y` coordinates are converted to the radial coordinate `r` and
    the angular coordinate `phi`.

    """
    r = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    return {"r": r, "phi": phi}, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CartesianPos2D],
    from_vector: type[PolarPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.OptAuxDict]:
    """PolarPos -> CartesianPos2D.

    The `r` and `phi` coordinates are converted to the `x` and `y` coordinates.

    """
    x = p["r"] * jnp.cos(p["phi"])
    y = p["r"] * jnp.sin(p["phi"])
    return {"x": x, "y": y}, combine_aux(in_aux, out_aux)


###############################################################################
# Corresponding Cartesian classes


@dispatch
def cartesian_vector_type(
    obj: type[AbstractPos2D] | AbstractPos2D, /
) -> type[CartesianPos2D]:
    """AbstractPos2D -> CartesianPos2D."""
    return CartesianPos2D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractVel2D] | AbstractVel2D, /
) -> type[CartesianVel2D]:
    """AbstractVel2D -> CartesianVel2D."""
    return CartesianVel2D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractAcc2D] | AbstractAcc2D, /
) -> type[CartesianAcc2D]:
    """AbstractPos -> CartesianAcc2D."""
    return CartesianAcc2D


###############################################################################
# Corresponding time derivative classes

# -----------------------------------------------
# Position -> Velocity


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianPos2D] | CartesianPos2D, /
) -> type[CartesianVel2D]:
    """Return the corresponding time derivative class."""
    return CartesianVel2D


@dispatch
def time_derivative_vector_type(obj: type[PolarPos] | PolarPos, /) -> type[PolarVel]:
    """Return the corresponding time derivative class."""
    return PolarVel


@dispatch
def time_derivative_vector_type(
    obj: type[TwoSpherePos] | TwoSpherePos, /
) -> type[TwoSphereVel]:
    """Return the corresponding time derivative class."""
    return TwoSphereVel


# -----------------------------------------------
# Velocity -> Position


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianVel2D] | CartesianVel2D, /
) -> type[CartesianPos2D]:
    """Return the corresponding time antiderivative class."""
    return CartesianPos2D


@dispatch
def time_antiderivative_vector_type(
    obj: type[PolarVel] | PolarVel, /
) -> type[PolarPos]:
    """Return the corresponding time antiderivative class."""
    return PolarPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[TwoSphereVel] | TwoSphereVel, /
) -> type[TwoSpherePos]:
    """Return the corresponding time antiderivative class."""
    return TwoSpherePos


# -----------------------------------------------
# Velocity -> Acceleration


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianVel2D] | CartesianVel2D, /
) -> type[CartesianAcc2D]:
    """Return the corresponding time derivative class."""
    return CartesianAcc2D


@dispatch
def time_derivative_vector_type(obj: type[PolarVel] | PolarVel, /) -> type[PolarAcc]:
    """Return the corresponding time derivative class."""
    return PolarAcc


@dispatch
def time_derivative_vector_type(
    obj: type[TwoSphereVel] | TwoSphereVel, /
) -> type[TwoSphereAcc]:
    """Return the corresponding time derivative class."""
    return TwoSphereAcc


# -----------------------------------------------
# Acceleration -> Velocity


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianAcc2D] | CartesianAcc2D, /
) -> type[CartesianVel2D]:
    """Return the corresponding time antiderivative class."""
    return CartesianVel2D


@dispatch
def time_antiderivative_vector_type(
    obj: type[PolarAcc] | PolarAcc, /
) -> type[PolarVel]:
    """Return the corresponding time antiderivative class."""
    return PolarVel


@dispatch
def time_antiderivative_vector_type(
    obj: type[TwoSphereAcc] | TwoSphereAcc, /
) -> type[TwoSphereVel]:
    """Return the corresponding time antiderivative class."""
    return TwoSphereVel
