# pylint: disable=duplicate-code
"""3-dimensional."""

__all__: list[str] = []

from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

from unxt import Quantity

from .builtin import Cartesian3DVector
from coordinax._typing import TimeBatchOrScalar
from coordinax.operators._base import AbstractOperator, op_call_dispatch

Q3: TypeAlias = Shaped[Quantity["length"], "*#batch 3"]


@op_call_dispatch
def call(self: AbstractOperator, x: Q3, /) -> Q3:
    """Dispatch to the operator's `__call__` method."""
    return self(Cartesian3DVector.constructor(x))


@op_call_dispatch
def call(
    self: AbstractOperator, x: Q3, t: TimeBatchOrScalar, /
) -> tuple[Q3, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslationOperator(Quantity([1, 2, 3], "kpc"))
    >>> op
    GalileanSpatialTranslationOperator( translation=Cartesian3DVector( ... ) )

    We can then apply the operator to a position:

    >>> pos = Quantity([1.0, 2.0, 3.0], "kpc")
    >>> t = Quantity(0.0, "Gyr")

    >>> op(pos, t)
    (Quantity['length'](Array([2., 4., 6.], dtype=float32), unit='kpc'),
        Quantity['time'](Array(0., dtype=float32, ...), unit='Gyr'))

    """
    vec, t = self(Cartesian3DVector.constructor(x), t)
    return convert(vec, Quantity), t
