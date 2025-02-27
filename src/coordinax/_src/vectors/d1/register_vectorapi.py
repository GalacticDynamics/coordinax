"""Representation of coordinates in different systems."""

__all__: list[str] = []


from plum import dispatch

import coordinax._src.vectors.custom_types as ct
from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from .cartesian import CartesianAcc1D, CartesianPos1D, CartesianVel1D
from .radial import RadialAcc, RadialPos, RadialVel
from coordinax._src.vectors.private_api import combine_aux

###############################################################################
# Vector Transformation

# =============================================================================
# `vconvert_impl`


@dispatch
def vconvert(
    to_vector: type[CartesianPos1D],
    from_vector: type[RadialPos],
    params: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """RadialPos -> CartesianPos1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.
    """
    return {"x": params["r"]}, combine_aux(in_aux, out_aux)


@dispatch
def vconvert(
    to_vector: type[RadialPos],
    from_vector: type[CartesianPos1D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos1D -> RadialPos.

    The `x` coordinate is converted to the `r` coordinate of the 1D system.
    """
    return {"r": p["x"]}, combine_aux(in_aux, out_aux)


###############################################################################
# Corresponding Cartesian classes


@dispatch
def cartesian_vector_type(
    obj: type[AbstractPos1D] | AbstractPos1D, /
) -> type[AbstractPos1D]:
    """Return the corresponding Cartesian vector class."""
    return CartesianPos1D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractVel1D] | AbstractVel1D, /
) -> type[AbstractVel1D]:
    """Return the corresponding Cartesian vector class."""
    return CartesianVel1D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractAcc1D] | AbstractAcc1D, /
) -> type[AbstractAcc1D]:
    """Return the corresponding Cartesian vector class."""
    return CartesianAcc1D


###############################################################################
# Corresponding time derivative classes

# -----------------------------------------------
# Position -> Velocity


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianPos1D] | CartesianPos1D, /
) -> type[CartesianVel1D]:
    """Return the corresponding time derivative class."""
    return CartesianVel1D


@dispatch
def time_derivative_vector_type(obj: type[RadialPos] | RadialPos, /) -> type[RadialVel]:
    """Return the corresponding time derivative class."""
    return RadialVel


# -----------------------------------------------
# Velocity -> Position


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianVel1D] | CartesianVel1D, /
) -> type[CartesianPos1D]:
    """Return the corresponding time antiderivative class."""
    return CartesianPos1D


@dispatch
def time_antiderivative_vector_type(
    obj: type[RadialVel] | RadialVel, /
) -> type[RadialPos]:
    """Return the corresponding time antiderivative class."""
    return RadialPos


# -----------------------------------------------
# Velocity -> Acceleration


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianVel1D] | CartesianVel1D, /
) -> type[CartesianAcc1D]:
    """Return the corresponding time derivative class."""
    return CartesianAcc1D


@dispatch
def time_derivative_vector_type(obj: type[RadialVel] | RadialVel, /) -> type[RadialAcc]:
    """Return the corresponding time derivative class."""
    return RadialAcc


# -----------------------------------------------
# Acceleration -> Velocity


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianAcc1D] | CartesianAcc1D, /
) -> type[CartesianVel1D]:
    """Return the corresponding time antiderivative class."""
    return CartesianVel1D


@dispatch
def time_antiderivative_vector_type(
    obj: type[RadialAcc] | RadialAcc, /
) -> type[RadialVel]:
    """Return the corresponding time antiderivative class."""
    return RadialVel
