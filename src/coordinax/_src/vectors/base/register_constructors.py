"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Mapping
from typing import Any

from jaxtyping import ArrayLike
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import AbstractQuantity

from .vector import AbstractVector
from coordinax._src.typing import Unit


@dispatch
def vector(obj: AbstractVector, /) -> AbstractVector:
    """Construct a vector from a vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> cx.vector(cart) is cart
    True

    """
    return obj


@dispatch
def vector(cls: type[AbstractVector], obj: Mapping[str, Any], /) -> AbstractVector:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Quantity(1, "m"), "y": u.Quantity(2, "m"),
    ...       "z": u.Quantity(3, "m")}
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    >>> xs = {"x": u.Quantity([1, 2], "m"), "y": u.Quantity([3, 4], "m"),
    ...       "z": u.Quantity([5, 6], "m")}
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [[1 3 5]
        [2 4 6]]>

    """
    return cls(**obj)


@dispatch
def vector(cls: type[AbstractVector], obj: AbstractQuantity, /) -> AbstractVector:
    """Construct a vector from a quantity.

    This will fail for most non-position vectors, except Cartesian vectors,
    since they generally do not have the same dimensions, nor can be converted
    from a Cartesian vector without additional information.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Mismatch:

    >>> try: cx.vecs.CartesianPos1D.from_(u.Quantity([1, 2, 3], "m"))
    ... except ValueError as e: print(e)
    Cannot construct <class 'coordinax...CartesianPos1D'> from 3 components.

    Pos 1D:

    >>> cx.vecs.CartesianPos1D.from_(u.Quantity(1, "meter"))
    CartesianPos1D( x=Quantity[...](value=...i32[], unit=Unit("m")) )

    >>> cx.vecs.CartesianPos1D.from_(u.Quantity([1], "meter"))
    CartesianPos1D(x=Quantity[...](value=i32[], unit=Unit("m")))

    >>> cx.vecs.CartesianPos1D.from_(cx.Distance(1, "meter"))
    CartesianPos1D( x=Quantity[...](value=...i32[], unit=Unit("m")) )

    >>> cx.vecs.RadialPos.from_(u.Quantity(1, "meter"))
    RadialPos(r=Distance(value=...i32[], unit=Unit("m")))

    >>> cx.vecs.RadialPos.from_(u.Quantity([1], "meter"))
    RadialPos(r=Distance(value=...i32[], unit=Unit("m")))

    Vel 1D:

    >>> cx.vecs.CartesianVel1D.from_(u.Quantity(1, "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.CartesianVel1D.from_(u.Quantity([1], "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.RadialVel.from_(u.Quantity(1, "m/s"))
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.RadialVel.from_(u.Quantity([1], "m/s"))
    RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    Acc 1D:

    >>> cx.vecs.CartesianAcc1D.from_(u.Quantity(1, "m/s2"))
    CartesianAcc1D( d2_x=... )

    >>> cx.vecs.CartesianAcc1D.from_(u.Quantity([1], "m/s2"))
    CartesianAcc1D( d2_x=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    >>> cx.vecs.RadialAcc.from_(u.Quantity(1, "m/s2"))
    RadialAcc( d2_r=... )

    >>> cx.vecs.RadialAcc.from_(u.Quantity([1], "m/s2"))
    RadialAcc( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    Pos 2D:

    >>> vec = cx.vecs.CartesianPos2D.from_(u.Quantity([1, 2], "m"))
    >>> vec
    CartesianPos2D(
        x=Quantity[...](value=i32[], unit=Unit("m")),
        y=Quantity[...](value=i32[], unit=Unit("m"))
    )

    Vel 2D:

    >>> vec = cx.vecs.CartesianVel2D.from_(u.Quantity([1, 2], "m/s"))
    >>> print(vec)
    <CartesianVel2D (d_x[m / s], d_y[m / s])
        [1 2]>

    Acc 2D:

    >>> vec = cx.vecs.CartesianAcc2D.from_(u.Quantity([1, 2], "m/s2"))
    >>> print(vec)
    <CartesianAcc2D (d2_x[m / s2], d2_y[m / s2])
        [1 2]>

    Pos 3D:

    >>> vec = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    Vel 3D:

    >>> vec = cx.CartesianVel3D.from_(u.Quantity([1, 2, 3], "m/s"))
    >>> print(vec)
    <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
        [1 2 3]>

    Acc 3D:

    >>> vec = cx.vecs.CartesianAcc3D.from_(u.Quantity([1, 2, 3], "m/s2"))
    >>> print(vec)
    <CartesianAcc3D (d2_x[m / s2], d2_y[m / s2], d2_z[m / s2])
        [1 2 3]>

    Generic 3D:

    >>> vec = cx.vecs.CartesianGeneric3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianGeneric3D (x[m], y[m], z[m])
        [1 2 3]>

    """
    # Ensure the object is at least 1D
    obj = jnp.atleast_1d(obj)

    # Check the dimensions
    if obj.shape[-1] != cls._dimensionality():
        msg = f"Cannot construct {cls} from {obj.shape[-1]} components."
        raise ValueError(msg)

    # Map the components
    comps = {k: obj[..., i] for i, k in enumerate(cls.components)}

    # Construct the vector from the mapping
    return cls.from_(comps)


@dispatch
def vector(
    cls: type[AbstractVector], obj: ArrayLike | list[Any], unit: Unit | str, /
) -> AbstractVector:
    """Construct a vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "meter")
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.CartesianPos3D.from_(xs, "meter")
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [[1 2 3]
        [4 5 6]]>

    """
    obj = u.Quantity.from_(jnp.asarray(obj), unit)
    return cls.from_(obj)  # re-dispatch


@dispatch
def vector(cls: type[AbstractVector], obj: AbstractVector, /) -> AbstractVector:
    """Construct a vector from another vector.

    Raises
    ------
    TypeError
        If the object is not an instance of the vector class.

    Parameters
    ----------
    cls : type[AbstractVector], positional-only
        The vector class.
    obj : :class:`coordinax.AbstractVector`, positional-only
        The vector to construct from.

    Examples
    --------
    >>> import coordinax as cx

    Positions:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")

    >>> cart = cx.CartesianPos3D.from_(q)
    >>> print(cart)
    <CartesianPos3D (x[km], y[km], z[km])
        [1 2 3]>

    >>> cx.vecs.AbstractPos3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalPos)
    >>> cx.vecs.AbstractPos3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalPos)
    >>> cx.vecs.AbstractPos3D.from_(cyl) is cyl
    True

    Velocities:

    >>> p = cx.CartesianVel3D.from_([1, 2, 3], "km/s")

    >>> cart = cx.CartesianVel3D.from_(p)
    >>> cx.vecs.AbstractVel3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(cyl) is cyl
    True

    Accelerations:

    >>> p = cx.CartesianVel3D.from_([1, 1, 1], "km/s")

    >>> cart = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> cx.vecs.AbstractAcc3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.vecs.SphericalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(cyl) is cyl
    True

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # Avoid copying if the types are the same. `isinstance` is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(field_items(obj)))
