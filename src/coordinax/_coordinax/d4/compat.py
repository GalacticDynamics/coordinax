"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from unxt import Quantity

from .spacetime import FourVector
from coordinax._coordinax.operators.base import AbstractOperator, op_call_dispatch

#####################################################################
# Convert to Quantity


@conversion_method(type_from=FourVector, type_to=Quantity)  # type: ignore[misc]
def vec_to_q(obj: FourVector, /) -> Shaped[Quantity["length"], "*batch 4"]:
    """`coordinax.AbstractPosition3D` -> `unxt.Quantity`."""
    cart = convert(obj.q, Quantity)
    return xp.concat([obj.c * obj.t[..., None], cart], axis=-1)


#####################################################################
# Operators


@op_call_dispatch
def call(self: AbstractOperator, v4: FourVector, /) -> FourVector:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslationOperator(Quantity([1, 2, 3], "kpc"))
    >>> op
    GalileanSpatialTranslationOperator( translation=CartesianPosition3D( ... ) )

    We can then apply the operator to a position:

    >>> pos = cx.FourVector.constructor([0, 1.0, 2.0, 3.0], "kpc")
    >>> pos
    FourVector( t=Quantity[PhysicalType('time')](...), q=CartesianPosition3D( ... ) )

    >>> newpos = op(pos)
    >>> newpos
    FourVector( t=Quantity[PhysicalType('time')](...), q=CartesianPosition3D( ... ) )
    >>> newpos.q.x
    Quantity['length'](Array(2., dtype=float32), unit='kpc')

    """
    q, t = self(v4.q, v4.t)
    return FourVector(t=t, q=q)


@op_call_dispatch
def call(
    self: AbstractOperator, x: Shaped[Quantity["length"], "*batch 4"], /
) -> Shaped[Quantity["length"], "*batch 4"]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslationOperator(Quantity([1, 2, 3], "kpc"))
    >>> op
    GalileanSpatialTranslationOperator( translation=CartesianPosition3D( ... ) )

    We can then apply the operator to a position:

    >>> pos = Quantity([0, 1.0, 2.0, 3.0], "kpc")
    >>> pos
    Quantity['length'](Array([0., 1., 2., 3.], dtype=float32), unit='kpc')

    >>> newpos = op(pos)
    >>> newpos
    Quantity['length'](Array([0., 2., 4., 6.], dtype=float32), unit='kpc')

    """
    return convert(self(FourVector.constructor(x)), Quantity)
