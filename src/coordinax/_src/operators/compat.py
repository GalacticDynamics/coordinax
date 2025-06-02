"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import Any, TypeAlias

from jaxtyping import Shaped
from plum import convert

import unxt as u
from dataclassish import replace

from .base import AbstractOperator
from .pipe import Pipe
from coordinax._src.custom_types import TimeBatchOrScalar
from coordinax._src.vectors.collection import Space
from coordinax._src.vectors.d1 import CartesianPos1D
from coordinax._src.vectors.d2 import CartesianPos2D
from coordinax._src.vectors.d3 import CartesianPos3D
from coordinax._src.vectors.d4 import FourVector

# ============================================================================
# 1-Dimensional

Q1: TypeAlias = Shaped[u.AbstractQuantity, "*#batch 1"]


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, x: Q1, /, **kwargs: Any) -> Q1:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1], "km")
    >>> q = u.Quantity([0], "km")
    >>> op(q)
    Quantity(Array([1], dtype=int32), unit='km')

    """
    # Quantity -> CartesianPos1D -> [Operator] -> Quantity
    return convert(self(CartesianPos1D.from_(x), **kwargs), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, t: TimeBatchOrScalar, x: Q1, /, **kwargs: Any
) -> tuple[TimeBatchOrScalar, Q1]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1], "km")
    >>> q = u.Quantity([0], "km")
    >>> t = u.Quantity(0, "s")
    >>> op(t, q)
    (Quantity(Array(0, dtype=int32, ...), unit='s'),
     Quantity(Array([1], dtype=int32), unit='km'))

    """
    t, vec = self(t, CartesianPos1D.from_(x), **kwargs)
    return t, convert(vec, u.Quantity)


# ============================================================================
# 2-Dimensional


Q2: TypeAlias = Shaped[u.AbstractQuantity, "*#batch 2"]


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, x: Q2, /, **kwargs: Any) -> Q2:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = u.Quantity([1, 2], "m")
    >>> op = cx.ops.GalileanSpatialTranslation(u.Quantity([-1, -1], "m"))
    >>> op(q)
    Quantity(Array([0, 1], dtype=int32), unit='m')

    """
    return convert(self(CartesianPos2D.from_(x), **kwargs), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, t: TimeBatchOrScalar, x: Q2, /, **kwargs: Any
) -> tuple[TimeBatchOrScalar, Q2]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = u.Quantity([1, 2], "m")
    >>> t = u.Quantity(0, "s")
    >>> op = cx.ops.GalileanSpatialTranslation(u.Quantity([-1, -1], "m"))
    >>> op(t, q)
    (Quantity(Array(0, dtype=int32, ...), unit='s'),
     Quantity(Array([0, 1], dtype=int32), unit='m'))

    """
    t, vec = self(t, CartesianPos2D.from_(x), **kwargs)
    return t, convert(vec, u.Quantity)


# ============================================================================
# 3-Dimensional


Q3: TypeAlias = Shaped[u.AbstractQuantity, "*#batch 3"]


@AbstractOperator.__call__.dispatch_multi(
    (AbstractOperator, Q3),
    (Pipe, Q3),
)
def call(self: AbstractOperator, q: Q3, /, **kwargs: Any) -> Q3:
    r"""Operate on a 3D Quantity.

    `q` is the position vector. This is interpreted as a 3D CartesianVector.
    See `coordinax.CartesianPos3D` for more details.

    Returns
    -------
    x' : Quantity['length', '*#batch 3']
        The operated-upon position vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> shift = u.Quantity([1.0, 2.0, 3.0], "km")
    >>> op = cx.ops.GalileanSpatialTranslation(shift)

    >>> q = u.Quantity([0.0, 0, 0], "km")
    >>> op(q)
    Quantity(Array([1., 2., 3.], dtype=float32), unit='km')

    """
    cart = CartesianPos3D.from_(q)
    result = self(cart, **kwargs)
    return convert(result.vconvert(CartesianPos3D), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, t: TimeBatchOrScalar, x: Q3, /, **kwargs: Any
) -> tuple[TimeBatchOrScalar, Q3]:
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

    >>> q = u.Quantity([1.0, 2.0, 3.0], "km")
    >>> t = u.Quantity(0.0, "Gyr")

    >>> op(t, q)
    (Quantity(Array(0., dtype=float32, ...), unit='Gyr'),
     Quantity(Array([2., 4., 6.], dtype=float32), unit='km'))

    """
    t, vec = self(t, CartesianPos3D.from_(x), **kwargs)
    return t, convert(vec, u.Quantity)


# ============================================================================
# 4-Dimensional


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, v4: FourVector, /, **kwargs: Any) -> FourVector:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
    >>> op
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    We can then apply the operator to a position:

    >>> pos = cx.FourVector.from_([0, 1.0, 2.0, 3.0], "km")
    >>> pos
    FourVector( t=Quantity(...), q=CartesianPos3D( ... ) )

    >>> newpos = op(pos)
    >>> newpos
    FourVector( t=Quantity(...), q=CartesianPos3D( ... ) )
    >>> newpos.q.x
    Quantity(Array(2., dtype=float32), unit='km')

    Now on a VelocityBoost:

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> v4 = cx.FourVector.from_([0, 0, 0, 0], "m")
    >>> newv4 = op(v4)
    >>> print(newv4)
    <FourVector: (t[m s / km], q=(x, y, z) [m])
        [0. 0. 0. 0.]>

    """
    t, q = self(v4.t, v4.q, **kwargs)
    return FourVector(t=t, q=q)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator,
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


# ============================================================================
# Space


@AbstractOperator.__call__.dispatch
def call(self: AbstractOperator, space: Space, /, **__: Any) -> Space:
    r"""Apply the boost to a Space.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    On positions:

    >>> op = cx.ops.GalileanRotation.from_([[0., -1, 0], [1, 0, 0], [0, 0, 1]])

    >>> x = cx.CartesianPos3D.from_([1., 2, 3], "m")
    >>> space = cx.Space(length=x)
    >>> new_space = op(space)  # no effect
    >>> print(new_space["length"])
    <CartesianPos3D: (x, y, z) [m]
        [-2.  1.  3.]>

    On positions and velocities:

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> v = cx.CartesianVel3D.from_([4., 5, 6], "m/s")
    >>> space = cx.Space(length=x, speed=v)
    >>> space
    Space({ 'length': CartesianPos3D( ... ), 'speed': CartesianVel3D( ... ) })

    >>> new_space = op(space)
    >>> new_space.keys()
    dict_keys(['length', 'speed'])
    >>> print(new_space["speed"])
    <CartesianVel3D: (x, y, z) [m / s]
        [5. 7. 9.]>

    """
    # TODO: figure out how to do this in general, not just for q &/ p
    if "length" not in space:
        raise NotImplementedError("TODO")  # noqa: EM101
    if "speed" in space and "acceleration" in space:
        q, p, a = self(space["length"], space["speed"], space["acceleration"])
        out = replace(space, length=q, speed=p, acceleration=a)
    elif "speed" in space:
        q, p = self(space["length"], space["speed"])
        out = replace(space, length=q, speed=p)
    else:
        out = replace(space, length=self(space["length"]))

    return out
