"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import NoReturn

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
    CartesianPosND(q=Quantity([1], unit='km'))

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1], "km"))
    CartesianPosND(q=Quantity([1], unit='km'))

    >>> cx.vecs.CartesianVelND.from_(u.Quantity(1, "km/s"))
    CartesianVelND(q=Quantity([1], unit='km / s'))

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1], "km/s"))
    CartesianVelND(q=Quantity([1], unit='km / s'))

    >>> cx.vecs.CartesianAccND.from_(u.Quantity(1, "km/s2"))
    CartesianAccND(q=Quantity([1], unit='km / s2'))

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1], "km/s2"))
    CartesianAccND(q=Quantity([1], unit='km / s2'))

    2D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2], "km"))
    CartesianPosND(q=Quantity([1, 2], unit='km'))

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2], "km/s"))
    CartesianVelND(q=Quantity([1, 2], unit='km / s'))

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2], "km/s2"))
    CartesianAccND(q=Quantity([1, 2], unit='km / s2'))

    3D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3], "km"))
    CartesianPosND(q=Quantity([1, 2, 3], unit='km'))

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3], "km/s"))
    CartesianVelND(q=Quantity([1, 2, 3], unit='km / s'))

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3], "km/s2"))
    CartesianAccND(q=Quantity([1, 2, 3], unit='km / s2'))

    4D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3, 4], "km"))
    CartesianPosND(q=Quantity([1, 2, 3, 4], unit='km'))

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3, 4], "km/s"))
    CartesianVelND(q=Quantity([1, 2, 3, 4], unit='km / s'))

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3, 4], "km/s2"))
    CartesianAccND(q=Quantity([1, 2, 3, 4], unit='km / s2'))

    """
    return cls(jnp.atleast_1d(x))


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
