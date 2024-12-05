"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import convert

import unxt as u

from .cartesian import CartesianPos3D
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.typing import TimeBatchOrScalar

#####################################################################
# Operators


Q3: TypeAlias = Shaped[u.Quantity["length"], "*#batch 3"]


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, q: Q3, /) -> Q3:
    r"""Operate on a 3D Quantity.

    `q` is the position vector. This is interpreted as a 3D CartesianVector.
    See :class:`coordinax.CartesianPos3D` for more details.

    Returns
    -------
    x' : Quantity['length', '*#batch 3']
        The operated-upon position vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.operators as cxo

    >>> shift = u.Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op = cxo.GalileanSpatialTranslation(shift)

    >>> q = u.Quantity([0.0, 0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    """
    cart = CartesianPos3D.from_(q)
    result = self(cart)
    return convert(result.represent_as(CartesianPos3D), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, x: Q3, t: TimeBatchOrScalar, /
) -> tuple[Q3, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.operators.GalileanSpatialTranslation.from_([1, 2, 3], "kpc")
    >>> op
    GalileanSpatialTranslation( translation=CartesianPos3D( ... ) )

    We can then apply the operator to a position:

    >>> q = u.Quantity([1.0, 2.0, 3.0], "kpc")
    >>> t = u.Quantity(0.0, "Gyr")

    >>> op(q, t)
    (Quantity['length'](Array([2., 4., 6.], dtype=float32), unit='kpc'),
     Quantity['time'](Array(0., dtype=float32, ...), unit='Gyr'))

    """
    vec, t = self(CartesianPos3D.from_(x), t)
    return convert(vec, u.Quantity), t
