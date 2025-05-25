"""Built-in 4-vector classes."""

__all__: list[str] = []

from dataclasses import replace
from typing import Any, cast

import equinox as eqx
from plum import dispatch

import unxt as u

from .spacetime import FourVector
from coordinax._src.vectors import api
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianPos3D


@dispatch
def vector(cls: type[FourVector], obj: u.AbstractQuantity, /) -> FourVector:
    """Construct a vector from a Quantity array.

    The ``Quantity[Any, (*#batch, 4), "..."]`` is expected to have the
    components as the last dimension. The 4 components are the (c x) time, x, y,
    z.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = u.Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
    >>> vec = cx.FourVector.from_(xs)
    >>> print(vec)
    <FourVector: (t[m s / km], q=(x, y, z) [m])
        [0. 1. 2. 3.]>

    >>> xs = u.Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
    >>> vec = cx.FourVector.from_(xs)
    >>> print(vec)
    <FourVector: (t[m s / km], q=(x, y, z) [m])
        [[0.000e+00 1.000e+00 2.000e+00 3.000e+00]
         [3.336e-05 4.000e+00 5.000e+00 6.000e+00]]>

    """
    _ = eqx.error_if(
        obj,
        obj.shape[-1] != 4,
        f"Cannot construct {cls} from array with shape {obj.shape}.",
    )
    c = cls.__dataclass_fields__["c"].default.default
    return cls(t=obj[..., 0] / c, q=obj[..., 1:], c=c)


# =============================================================================


@dispatch
def vconvert(
    spatial_target: type[AbstractPos3D], current: FourVector, /, **kwargs: Any
) -> FourVector:
    """Convert the spatial part of a 4-vector to a different 3-vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(cx.vconvert(cx.vecs.CylindricalPos, w))
    <FourVector: (t[s], q=(rho[m], phi[rad], z[m]))
        [1.    2.236 1.107 3.   ]>

    """
    q = cast(AbstractPos3D, api.vconvert(spatial_target, current.q, **kwargs))
    return replace(current, q=q)


# =============================================================================


@dispatch
def spatial_component(x: FourVector, /) -> AbstractPos3D:
    """Return the spatial component of the vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(spatial_component(w))
    <CartesianPos3D: (x, y, z) [m]
        [1 2 3]>

    """
    return x.q


# =============================================================================
# Corresponding Cartesian class


@dispatch
def cartesian_vector_type(
    obj: type[FourVector] | FourVector, /
) -> type[CartesianPos3D]:
    """Return the corresponding Cartesian vector class."""
    return CartesianPos3D
