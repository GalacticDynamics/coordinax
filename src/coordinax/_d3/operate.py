# pylint: disable=duplicate-code
"""3-dimensional."""

__all__: list[str] = []

from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

from unxt import Quantity

from .cartesian import CartesianPosition3D
from coordinax._typing import TimeBatchOrScalar
from coordinax.operators._base import AbstractOperator, op_call_dispatch

Q3: TypeAlias = Shaped[Quantity["length"], "*#batch 3"]


@op_call_dispatch
def call(self: AbstractOperator, q: Q3, /) -> Q3:
    r"""Operate on a 3D Quantity.

    `q` is the position vector. This is interpreted as a 3D CartesianVector.
    See :class:`coordinax.CartesianPosition3D` for more details.

    Returns
    -------
    x' : Quantity['length', '*#batch 3']
        The operated-upon position vector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import coordinax.operators as cxo

    >>> shift = Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op = cxo.GalileanSpatialTranslationOperator(shift)

    >>> q = Quantity([0.0, 0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    Since :meth:`AbstractOperator.__call__` uses multiple dispatch, we
    explicitly call this registered method.

    >>> op.__call__._f.resolve_method((op, q))[0](op, q)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    """
    return convert(
        self(CartesianPosition3D.constructor(q)).represent_as(CartesianPosition3D),
        Quantity,
    )


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
    GalileanSpatialTranslationOperator( translation=CartesianPosition3D( ... ) )

    We can then apply the operator to a position:

    >>> q = Quantity([1.0, 2.0, 3.0], "kpc")
    >>> t = Quantity(0.0, "Gyr")

    >>> op(q, t)
    (Quantity['length'](Array([2., 4., 6.], dtype=float32), unit='kpc'),
        Quantity['time'](Array(0., dtype=float32, ...), unit='Gyr'))

    Since :meth:`AbstractOperator.__call__` uses multiple dispatch, we
    explicitly call this registered method.

    >>> op.__call__._f.resolve_method((op, q, t))[0](op, q, t)
    (Quantity['length'](Array([2., 4., 6.], dtype=float32), unit='kpc'),
        Quantity['time'](Array(0., dtype=float32, ...), unit='Gyr'))

    """
    vec, t = self(CartesianPosition3D.constructor(x), t)
    return convert(vec, Quantity), t
