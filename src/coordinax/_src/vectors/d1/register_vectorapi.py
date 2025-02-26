"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import coordinax._src.vectors.custom_types as ct
from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from .cartesian import CartesianAcc1D, CartesianPos1D, CartesianVel1D
from .radial import RadialAcc, RadialPos, RadialVel
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel
from coordinax._src.vectors.private_api import combine_aux

###############################################################################
# Vector Transformation

# =============================================================================
# `vconvert_impl`


@dispatch
def vconvert_impl(
    to_vector: type[AbstractPos1D],
    from_vector: type[AbstractPos1D],
    params: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """AbstractPos -> CartesianPos1D -> AbstractPos."""
    params, aux = vconvert_impl(
        CartesianPos1D, from_vector, params, in_aux=in_aux, out_aux=None, units=units
    )
    params, aux = vconvert_impl(
        to_vector, CartesianPos1D, params, in_aux=aux, out_aux=out_aux, units=units
    )
    return params, aux


@dispatch.multi(
    # Positions
    (type[CartesianPos1D], type[CartesianPos1D], ct.ParamsDict),
    (type[RadialPos], type[RadialPos], ct.ParamsDict),
    # Velocities
    (type[CartesianVel1D], type[CartesianVel1D], ct.ParamsDict),
    (type[RadialVel], type[RadialVel], ct.ParamsDict),
    # Accelerations
    (type[CartesianAcc1D], type[CartesianAcc1D], ct.ParamsDict),
    (type[RadialAcc], type[RadialAcc], ct.ParamsDict),
)
def vconvert_impl(
    to_vector: type[AbstractVector],
    from_vector: type[AbstractVector],
    params: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """1D self transform."""
    return params, combine_aux(in_aux, out_aux)


@dispatch
def vconvert_impl(
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
def vconvert_impl(
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


# =============================================================================


@dispatch.multi(  # TODO: is the precedence needed?
    # Positions
    (type[CartesianPos1D], CartesianPos1D),
    (type[RadialPos], RadialPos),
    # Velocities
    (type[CartesianVel1D], CartesianVel1D, AbstractPos),
    (type[RadialVel], RadialVel, AbstractPos),
    (type[CartesianVel1D], CartesianVel1D),  # q not needed
    (type[RadialVel], RadialVel),  # q not needed
    # Accelerations
    (type[CartesianAcc1D], CartesianAcc1D, AbstractVel, AbstractPos),
    (type[RadialAcc], RadialAcc, AbstractVel, AbstractPos),
    (type[CartesianAcc1D], CartesianAcc1D),  # q,p not needed
    (type[RadialAcc], RadialAcc),  # q,p not needed
)
def vconvert(
    target: type[AbstractVector], current: AbstractVector, /, *args: Any, **kw: Any
) -> AbstractVector:
    """Self transform of 1D vectors.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos1D.from_(1, "m")
    >>> cx.vconvert(cx.vecs.CartesianPos1D, q) is q
    True

    >>> q = cx.vecs.RadialPos.from_(1, "m")
    >>> cx.vconvert(cx.vecs.RadialPos, q) is q
    True

    >>> p = cx.vecs.CartesianVel1D.from_(1, "m/s")
    >>> cx.vconvert(cx.vecs.CartesianVel1D, p) is p
    True
    >>> cx.vconvert(cx.vecs.CartesianVel1D, p, q) is p
    True

    >>> p = cx.vecs.RadialVel.from_(1, "m/s")
    >>> cx.vconvert(cx.vecs.RadialVel, p) is p
    True
    >>> cx.vconvert(cx.vecs.RadialVel, p, q) is p
    True

    >>> a = cx.vecs.CartesianAcc1D.from_(1, "m/s2")
    >>> cx.vconvert(cx.vecs.CartesianAcc1D, a) is a
    True
    >>> cx.vconvert(cx.vecs.CartesianAcc1D, a, p, q) is a
    True

    >>> a = cx.vecs.RadialAcc.from_(1, "m/s2")
    >>> cx.vconvert(cx.vecs.RadialAcc, a) is a
    True
    >>> cx.vconvert(cx.vecs.RadialAcc, a, p, q) is a
    True

    """
    return current


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
