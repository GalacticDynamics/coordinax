"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any, NoReturn

from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import AbstractQuantity

from .base import AbstractAccND, AbstractPosND, AbstractVelND
from .cartesian import CartesianAccND, CartesianPosND, CartesianVelND
from .poincare import PoincarePolarVector

###############################################################################


@dispatch
def vector(
    cls: type[CartesianPosND] | type[CartesianVelND] | type[CartesianAccND],
    x: AbstractQuantity,
    /,
) -> CartesianPosND | CartesianVelND | CartesianAccND:
    """Construct an N-dimensional acceleration.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    1D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity(1, "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[1], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[1], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity(1, "km/s"))
    CartesianVelND(
      q=Quantity[...]( value=...i32[1], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1], "km/s"))
    CartesianVelND(
      q=Quantity[...]( value=...i32[1], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity(1, "km/s2"))
    CartesianAccND(
      q=Quantity[...]( value=...i32[1], unit=Unit("km / s2") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1], "km/s2"))
    CartesianAccND(
      q=Quantity[...](value=i32[1], unit=Unit("km / s2"))
    )

    2D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[2], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2], "km/s"))
    CartesianVelND(
      q=Quantity[...]( value=...i32[2], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2], "km/s2"))
    CartesianAccND(
      q=Quantity[...](value=i32[2], unit=Unit("km / s2"))
    )

    3D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[3], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3], "km/s"))
    CartesianVelND(
      q=Quantity[...]( value=...i32[3], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3], "km/s2"))
    CartesianAccND(
      q=Quantity[...](value=i32[3], unit=Unit("km / s2"))
    )

    4D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3, 4], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[4], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3, 4], "km/s"))
    CartesianVelND(
      q=Quantity[...]( value=...i32[4], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3, 4], "km/s2"))
    CartesianAccND(
      q=Quantity[...](value=i32[4], unit=Unit("km / s2"))
    )

    """
    return cls(jnp.atleast_1d(x))


###############################################################################
# Vector Conversion


@dispatch
def vconvert(
    target: type[CartesianPosND], current: CartesianPosND, /, **kwargs: Any
) -> CartesianPosND:
    """CartesianPosND -> CartesianPosND.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4], "km"))

    >>> cx.vconvert(cx.vecs.CartesianPosND, q) is q
    True

    """
    return current


@dispatch
def vconvert(
    target: type[CartesianVelND],
    current: CartesianVelND,
    position: CartesianPosND,
    /,
    **kwargs: Any,
) -> CartesianVelND:
    """CartesianVelND -> CartesianVelND.

    Examples
    --------
    >>> import coordinax as cx

    >>> x = cx.vecs.CartesianPosND.from_([1, 2, 3, 4], "km")
    >>> v = cx.vecs.CartesianVelND.from_([1, 2, 3, 4], "km/s")

    >>> cx.vconvert(cx.vecs.CartesianVelND, v, x) is v
    True

    """
    return current


# =============================================================================
# CartesianVelND


@dispatch
def vconvert(
    target: type[CartesianVelND], current: CartesianVelND, /
) -> CartesianVelND:
    """CartesianVelND -> CartesianVelND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.vecs.CartesianVelND.from_([1, 1, 1], "m/s")
    >>> cx.vconvert(cx.vecs.CartesianVelND, v) is v
    True

    """
    return current


# =============================================================================
# CartesianAccND


@dispatch
def vconvert(
    target: type[CartesianAccND], current: CartesianAccND, /
) -> CartesianAccND:
    """CartesianAccND -> CartesianAccND with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.vecs.CartesianAccND.from_([1, 1, 1], "m/s2")
    >>> cx.vconvert(cx.vecs.CartesianAccND, a) is a
    True

    """
    return current


# =============================================================================
# Poincare


@dispatch
def vconvert(
    target: type[PoincarePolarVector], current: PoincarePolarVector, /, **kwargs: Any
) -> PoincarePolarVector:
    """PoincarePolarVector -> PoincarePolarVector."""
    return current


###############################################################################
# Corresponding Cartesian class


@dispatch
def cartesian_vector_type(
    obj: type[AbstractPosND] | AbstractPosND, /
) -> type[CartesianPosND]:
    """Return the corresponding Cartesian vector class."""
    return CartesianPosND


@dispatch
def cartesian_vector_type(
    obj: type[AbstractVelND] | AbstractVelND, /
) -> type[CartesianVelND]:
    """Return the corresponding Cartesian vector class."""
    return CartesianVelND


@dispatch
def cartesian_vector_type(
    obj: type[AbstractAccND] | AbstractAccND, /
) -> type[CartesianAccND]:
    """Return the corresponding Cartesian vector class."""
    return CartesianAccND


@dispatch
def cartesian_vector_type(
    obj: type[PoincarePolarVector] | PoincarePolarVector, /
) -> NoReturn:
    """Return the corresponding Cartesian vector class.

    Examples
    --------
    >>> import coordinax as cx
    >>> try: cx.vecs.cartesian_vector_type(cx.vecs.PoincarePolarVector)
    ... except NotImplementedError as e:
    ...     print(e)
    PoincarePolarVector does not have a corresponding Cartesian class.

    """
    msg = "PoincarePolarVector does not have a corresponding Cartesian class."
    raise NotImplementedError(msg)


###############################################################################
# Corresponding time derivative classes

# -----------------------------------------------
# Position -> Velocity


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianPosND] | CartesianPosND, /
) -> type[CartesianVelND]:
    """Return the corresponding time derivative class."""
    return CartesianVelND


# -----------------------------------------------
# Velocity -> Position


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianVelND] | CartesianVelND, /
) -> type[CartesianPosND]:
    """Return the corresponding time antiderivative class."""
    return CartesianPosND


# -----------------------------------------------
# Velocity -> Acceleration


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianVelND] | CartesianVelND, /
) -> type[CartesianAccND]:
    """Return the corresponding time derivative class."""
    return CartesianAccND


# -----------------------------------------------
# Acceleration -> Velocity


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianAccND] | CartesianAccND, /
) -> type[CartesianVelND]:
    """Return the corresponding time antiderivative class."""
    return CartesianVelND
