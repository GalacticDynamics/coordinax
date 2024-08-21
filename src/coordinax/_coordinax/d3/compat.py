"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from dataclassish import field_values
from unxt import Quantity

from .cartesian import CartesianAcceleration3D, CartesianPosition3D, CartesianVelocity3D
from coordinax._coordinax.operators.base import AbstractOperator, op_call_dispatch
from coordinax._coordinax.typing import TimeBatchOrScalar
from coordinax._coordinax.utils import full_shaped

#####################################################################
# Convert to Quantity


@conversion_method(CartesianAcceleration3D, Quantity)  # type: ignore[misc]
@conversion_method(CartesianVelocity3D, Quantity)  # type: ignore[misc]
def vec_diff_to_q(obj: CartesianVelocity3D, /) -> Shaped[Quantity["speed"], "*batch 3"]:
    """`coordinax.CartesianVelocity3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> dif = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
    >>> convert(dif, Quantity)
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')

    >>> dif2 = cx.CartesianAcceleration3D.constructor([1, 2, 3], "km/s2")
    >>> convert(dif2, Quantity)
    Quantity['acceleration'](Array([1., 2., 3.], dtype=float32), unit='km / s2')

    """
    return xp.stack(tuple(field_values(full_shaped(obj))), axis=-1)


#####################################################################
# Operators


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
    cart = CartesianPosition3D.constructor(q)
    result = self(cart)
    return convert(result.represent_as(CartesianPosition3D), Quantity)


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
