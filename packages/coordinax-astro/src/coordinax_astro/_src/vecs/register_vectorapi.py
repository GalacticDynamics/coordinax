"""Built-in 4-vector classes."""

__all__: list[str] = []

from dataclasses import replace
from typing import Any, cast

import equinox as eqx
from jaxtyping import Shaped
from plum import convert, dispatch

import unxt as u

import coordinax.ops as cxo
import coordinax.vecs as cxv
from .spacetime import FourVector


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
    >>> from coordinax_astro import FourVector

    >>> xs = u.Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
    >>> vec = FourVector.from_(xs)
    >>> print(vec)
    <FourVector: (t[m s / km], q=(x, y, z) [m])
        [0. 1. 2. 3.]>

    >>> xs = u.Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
    >>> vec = FourVector.from_(xs)
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
    spatial_target: type[cxv.AbstractPos3D], current: FourVector, /, **kwargs: Any
) -> FourVector:
    """Convert the spatial part of a 4-vector to a different 3-vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> from coordinax_astro import FourVector

    >>> w = FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(cx.vconvert(cx.vecs.CylindricalPos, w))
    <FourVector: (t[s], q=(rho[m], phi[rad], z[m]))
        [1.    2.236 1.107 3.   ]>

    """
    q = cast(cxv.AbstractPos3D, cxv.vconvert(spatial_target, current.q, **kwargs))
    return replace(current, q=q)


@dispatch.multi(  # TODO: is the precedence needed?
    (type[FourVector], FourVector),
)
def vconvert(
    target: type[cxv.AbstractVector],
    current: cxv.AbstractVector,
    /,
    *args: Any,
    **kw: Any,
) -> cxv.AbstractVector:
    """Self transforms."""
    return current


# =============================================================================


@dispatch
def spatial_component(x: FourVector, /) -> cxv.AbstractPos3D:
    """Return the spatial component of the vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
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
) -> type[cxv.CartesianPos3D]:
    """Return the corresponding Cartesian vector class."""
    return cxv.CartesianPos3D


# =============================================================================
# Operators


@cxo.AbstractOperator.__call__.dispatch
def call(self: cxo.AbstractOperator, v4: FourVector, /, **kwargs: Any) -> FourVector:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import coordinax as cx
    >>> from coordinax_astro import FourVector

    We can then create a spatial translation operator:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
    >>> op
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    We can then apply the operator to a position:

    >>> pos = FourVector.from_([0, 1.0, 2.0, 3.0], "km")
    >>> pos
    FourVector( t=Quantity(...), q=CartesianPos3D( ... ) )

    >>> newpos = op(pos)
    >>> newpos
    FourVector( t=Quantity(...), q=CartesianPos3D( ... ) )
    >>> newpos.q.x
    Quantity(Array(2., dtype=float32), unit='km')

    Now on a VelocityBoost:

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> v4 = FourVector.from_([0, 0, 0, 0], "m")
    >>> newv4 = op(v4)
    >>> print(newv4)
    <FourVector: (t[m s / km], q=(x, y, z) [m])
        [0. 0. 0. 0.]>

    """
    t, q = self(v4.t, v4.q, **kwargs)
    return FourVector(t=t, q=q)


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.AbstractOperator,
    x: Shaped[u.Quantity["length"], "*batch 4"],
    /,
    **kwargs: Any,
) -> Shaped[u.Quantity["length"], "*batch 4"]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
    >>> op
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    We can then apply the operator to a position:

    >>> pos = u.Quantity([0, 1.0, 2.0, 3.0], "km")
    >>> pos
    Quantity(Array([0., 1., 2., 3.], dtype=float32), unit='km')

    >>> newpos = op(pos)
    >>> newpos
    Quantity(Array([0., 2., 4., 6.], dtype=float32), unit='km')

    """
    q4 = FourVector.from_(x)
    return convert(self(q4, **kwargs), u.Quantity)


@cxo.AbstractOperator.__call__.dispatch
def call(self: cxo.GalileanBoost, v4: FourVector, /, **__: Any) -> FourVector:
    r"""Apply the boost to the coordinates.

    Recall that this is spatial-only, the time is invariant.

    The operation is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, \mathbf{x} + \mathbf{v} t)

    Examples
    --------
    >>> import unxt as u

    """
    t, q = self(v4.t, v4.q)
    return replace(v4, t=t, q=q)


# ---------------------------


@cxo.AbstractOperator.__call__.dispatch
def call(
    self: cxo.GalileanSpatialTranslation, v4: FourVector, /, **__: Any
) -> cxv.AbstractPos:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_astro as cxa

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

    >>> v4 = cxa.FourVector.from_([0, 1, 2, 3], "km")
    >>> print(op(v4))
    <FourVector: (t[s], q=(x, y, z) [km])
        [0. 2. 3. 4.]>

    """
    return replace(v4, q=v4.q + self.delta_q)


@cxo.AbstractOperator.__call__.dispatch
def call(self: cxo.GalileanTranslation, x: FourVector, /, **__: Any) -> FourVector:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.ops as cxo
    >>> from coordinax_astro import FourVector

    Explicitly construct the translation operator:

    >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
    >>> shift = FourVector(u.Quantity(1, "Gyr"), qshift)
    >>> op = cx.ops.GalileanTranslation.from_(shift)

    Construct a vector to translate, using the convenience from_ (the
    0th component is :math:`c * t`, the rest are spatial components):

    >>> w = FourVector.from_([0, 1, 2, 3], "km")
    >>> w.t
    Quantity(Array(0., dtype=float32, ...), unit='s')

    Apply the translation operator:

    >>> new = op(w)
    >>> new.x
    Quantity(Array(2, dtype=int32), unit='km')

    >>> new.t.uconvert("Gyr")
    Quantity(Array(1., dtype=float32, ...), unit='Gyr')

    """
    return x + FourVector(self.delta_t, self.delta_q)


@cxo.GalileanTranslation.from_.dispatch  # type: ignore[misc]
def from_(
    cls: type[cxo.GalileanTranslation], shift: FourVector, /
) -> cxo.GalileanTranslation:
    """Construct a Galilean translation operator from a 4-vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> from coordinax_astro import FourVector

    >>> shift = FourVector.from_([0, 1, 2, 3], "km")
    >>> op = cx.ops.GalileanTranslation.from_(shift)
    >>> print(op)
    GalileanTranslation( ... )

    """
    return cxo.GalileanTranslation(
        delta_t=shift.t,
        delta_q=shift.q,
        # c=shift.c,
    )
