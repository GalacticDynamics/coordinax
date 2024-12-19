"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import Any, TypeAlias

from jaxtyping import Shaped
from plum import convert

import unxt as u
from dataclassish import replace

from .base import AbstractOperator
from .pipe import Pipe
from coordinax._src.typing import TimeBatchOrScalar
from coordinax._src.vectors.d1 import CartesianPos1D
from coordinax._src.vectors.d2 import CartesianPos2D
from coordinax._src.vectors.d3 import CartesianPos3D
from coordinax._src.vectors.d4 import FourVector
from coordinax._src.vectors.space import Space

# ============================================================================
# 1-Dimensional

Q1: TypeAlias = Shaped[u.Quantity["length"], "*#batch 1"]


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
    Quantity['length'](Array([1], dtype=int32), unit='km')

    """
    # Quantity -> CartesianPos1D -> [Operator] -> Quantity
    return convert(self(CartesianPos1D.from_(x), **kwargs), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, x: Q1, t: TimeBatchOrScalar, /, **kwargs: Any
) -> tuple[Q1, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1], "km")
    >>> q = u.Quantity([0], "km")
    >>> op(q, u.Quantity(0, "s"))
    (Quantity['length'](Array([1], dtype=int32), unit='km'),
     Quantity['time'](Array(0, dtype=int32, ...), unit='s'))

    """
    vec, t = self(CartesianPos1D.from_(x), t, **kwargs)
    return convert(vec, u.Quantity), t


# ============================================================================
# 2-Dimensional


Q2: TypeAlias = Shaped[u.Quantity["length"], "*#batch 2"]


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
    Quantity['length'](Array([0, 1], dtype=int32), unit='m')

    """
    return convert(self(CartesianPos2D.from_(x), **kwargs), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, x: Q2, t: TimeBatchOrScalar, /, **kwargs: Any
) -> tuple[Q2, TimeBatchOrScalar]:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = u.Quantity([1, 2], "m")
    >>> op = cx.ops.GalileanSpatialTranslation(u.Quantity([-1, -1], "m"))
    >>> op(q, u.Quantity(0, "s"))
    (Quantity['length'](Array([0, 1], dtype=int32), unit='m'),
     Quantity['time'](Array(0, dtype=int32, ...), unit='s'))

    """
    vec, t = self(CartesianPos2D.from_(x), t, **kwargs)
    return convert(vec, u.Quantity), t


# ============================================================================
# 3-Dimensional


Q3: TypeAlias = Shaped[u.Quantity["length"], "*#batch 3"]


@AbstractOperator.__call__.dispatch_multi(
    (AbstractOperator, Q3),
    (Pipe, Q3),
)
def call(self: AbstractOperator, q: Q3, /, **kwargs: Any) -> Q3:
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

    >>> shift = u.Quantity([1.0, 2.0, 3.0], "km")
    >>> op = cx.ops.GalileanSpatialTranslation(shift)

    >>> q = u.Quantity([0.0, 0, 0], "km")
    >>> op(q)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    """
    cart = CartesianPos3D.from_(q)
    result = self(cart, **kwargs)
    return convert(result.vconvert(CartesianPos3D), u.Quantity)


@AbstractOperator.__call__.dispatch
def call(
    self: AbstractOperator, x: Q3, t: TimeBatchOrScalar, /, **kwargs: Any
) -> tuple[Q3, TimeBatchOrScalar]:
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

    >>> op(q, t)
    (Quantity['length'](Array([2., 4., 6.], dtype=float32), unit='km'),
     Quantity['time'](Array(0., dtype=float32, ...), unit='Gyr'))

    """
    vec, t = self(CartesianPos3D.from_(x), t, **kwargs)
    return convert(vec, u.Quantity), t


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
    FourVector( t=Quantity[PhysicalType('time')](...), q=CartesianPos3D( ... ) )

    >>> newpos = op(pos)
    >>> newpos
    FourVector( t=Quantity[PhysicalType('time')](...), q=CartesianPos3D( ... ) )
    >>> newpos.q.x
    Quantity['length'](Array(2., dtype=float32), unit='km')

    Now on a VelocityBoost:

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> v4 = cx.FourVector.from_([0, 0, 0, 0], "m")
    >>> newv4 = op(v4)
    >>> print(newv4)
    <FourVector (t[m s / km], q=(x[m], y[m], z[m]))
        [0. 0. 0. 0.]>

    """
    q, t = self(v4.q, v4.t, **kwargs)
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
    Quantity['length'](Array([0., 1., 2., 3.], dtype=float32), unit='km')

    >>> newpos = op(pos)
    >>> newpos
    Quantity['length'](Array([0., 2., 4., 6.], dtype=float32), unit='km')

    """
    return convert(self(FourVector.from_(x), **kwargs), u.Quantity)


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
    <CartesianPos3D (x[m], y[m], z[m])
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
    <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
        [5. 7. 9.]>

    """
    # TODO: figure out how to do this in general, not just for q &/ p
    if "length" in space and "speed" in space:
        q, p = self(space["length"], space["speed"])
        out = replace(space, length=q, speed=p)

    else:
        out = replace(space, length=self(space["length"]))

    return out
