"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

import unxt as u

from .cartesian import CartesianPos2D
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.typing import TimeBatchOrScalar

Q2: TypeAlias = Shaped[u.Quantity["length"], "*#batch 2"]


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, x: Q2, /) -> Q2:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.operators as cxo

    >>> q = u.Quantity([1, 2], "m")
    >>> op = cxo.GalileanSpatialTranslationOperator(u.Quantity([-1, -1], "m"))
    >>> op(q)
    Quantity['length'](Array([0., 1.], dtype=float32), unit='m')

    """
    return convert(self(CartesianPos2D.from_(x)), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, x: Q2, t: TimeBatchOrScalar, /
) -> tuple[Q2, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.operators as cxo

    >>> q = u.Quantity([1, 2], "m")
    >>> op = cxo.GalileanSpatialTranslationOperator(u.Quantity([-1, -1], "m"))
    >>> op(q, u.Quantity(0, "s"))
    (Quantity['length'](Array([0., 1.], dtype=float32), unit='m'),
     Quantity['time'](Array(0, dtype=int32, ...), unit='s'))

    """
    vec, t = self(CartesianPos2D.from_(x), t)
    return convert(vec, u.Quantity), t
