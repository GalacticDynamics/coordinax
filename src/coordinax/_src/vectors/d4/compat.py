"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.numpy as jnp
import unxt as u

from .spacetime import FourVector
from coordinax._src.operators.base import AbstractOperator

#####################################################################
# Convert to Quantity


@conversion_method(type_from=FourVector, type_to=u.Quantity)  # type: ignore[misc]
def vec_to_q(obj: FourVector, /) -> Shaped[u.Quantity["length"], "*batch 4"]:
    """`coordinax.AbstractPos3D` -> `unxt.Quantity`."""
    cart = convert(obj.q, u.Quantity)
    return jnp.concat([obj.c * obj.t[..., None], cart], axis=-1)


#####################################################################
# Operators


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, v4: FourVector, /) -> FourVector:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslationOperator.from_([1, 2, 3], "kpc")
    >>> op
    GalileanSpatialTranslationOperator( translation=CartesianPos3D( ... ) )

    We can then apply the operator to a position:

    >>> pos = cx.FourVector.from_([0, 1.0, 2.0, 3.0], "kpc")
    >>> pos
    FourVector( t=Quantity[PhysicalType('time')](...), q=CartesianPos3D( ... ) )

    >>> newpos = op(pos)
    >>> newpos
    FourVector( t=Quantity[PhysicalType('time')](...), q=CartesianPos3D( ... ) )
    >>> newpos.q.x
    Quantity['length'](Array(2., dtype=float32), unit='kpc')

    """
    q, t = self(v4.q, v4.t)
    return FourVector(t=t, q=q)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, x: Shaped[u.Quantity["length"], "*batch 4"], /
) -> Shaped[u.Quantity["length"], "*batch 4"]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslationOperator.from_([1, 2, 3], "kpc")
    >>> op
    GalileanSpatialTranslationOperator( translation=CartesianPos3D( ... ) )

    We can then apply the operator to a position:

    >>> pos = u.Quantity([0, 1.0, 2.0, 3.0], "kpc")
    >>> pos
    Quantity['length'](Array([0., 1., 2., 3.], dtype=float32), unit='kpc')

    >>> newpos = op(pos)
    >>> newpos
    Quantity['length'](Array([0., 2., 4., 6.], dtype=float32), unit='kpc')

    """
    return convert(self(FourVector.from_(x)), u.Quantity)
