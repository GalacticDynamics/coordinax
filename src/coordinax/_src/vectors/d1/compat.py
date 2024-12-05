"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

import unxt as u

from .cartesian import CartesianPos1D
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.typing import TimeBatchOrScalar

Q1: TypeAlias = Shaped[u.Quantity["length"], "*#batch 1"]


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, x: Q1, /) -> Q1:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.operators.GalileanSpatialTranslation.from_([1], "kpc")
    >>> q = u.Quantity([0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1.], dtype=float32), unit='kpc')

    """
    # Quantity -> CartesianPos1D -> [Operator] -> Quantity
    return convert(self(CartesianPos1D.from_(x)), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, x: Q1, t: TimeBatchOrScalar, /
) -> tuple[Q1, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.operators.GalileanSpatialTranslation.from_([1], "kpc")
    >>> q = u.Quantity([0], "kpc")
    >>> op(q, u.Quantity(0, "s"))
    (Quantity['length'](Array([1.], dtype=float32), unit='kpc'),
     Quantity['time'](Array(0, dtype=int32, ...), unit='s'))

    """
    vec, t = self(CartesianPos1D.from_(x), t)
    return convert(vec, u.Quantity), t
