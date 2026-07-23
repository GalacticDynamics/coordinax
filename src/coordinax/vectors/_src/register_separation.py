"""Separation (distance) between vectors.

Two metrics, modelled on `astropy.coordinates.SkyCoord.separation` /
``separation_3d``:

- :func:`separation_3d` -- the straight-line (Euclidean, embedded-Cartesian)
  distance between two points, returned as a `Distance`.
- :func:`separation` -- the angular separation between the two directions as
  seen from the origin, returned as an `Angle`.

Both are chart- and unit-invariant: the operands are brought into a common
Cartesian chart before measuring.  They are *frame-strict* -- coordinates in
different frames describe different physical points, so a cross-frame separation
is undefined and raises; align the operands with `to_frame` first.
"""

__all__ = ("separation", "separation_3d")

from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractVector
from coordinax.angles import Angle
from coordinax.distances import Distance

_LENGTH = u.dimension("length")


def _cartesian_components(
    a: AbstractVector, b: AbstractVector
) -> tuple[u.AbstractQuantity, u.AbstractQuantity]:
    """Return ``a`` and ``b`` as Cartesian `unxt.Quantity` vectors.

    Guards that the two vectors share a frame and a Cartesian chart, then
    converts each to a single `unxt.Quantity` (components stacked on the last
    axis) via the registered ``plum`` conversion.  Unit-aware `unxt.Quantity`
    arithmetic then makes the downstream measures chart- and unit-invariant with
    no manual per-component unit handling.
    """
    if a.frame != b.frame:
        msg = "cannot measure separation between vectors in different frames"
        raise ValueError(msg)

    ac = a.to_cartesian()
    bc = b.to_cartesian()
    if ac.chart != bc.chart:
        msg = "cannot measure separation between vectors on different manifolds"
        raise ValueError(msg)

    return convert(ac, u.Quantity), convert(bc, u.Quantity)


@dispatch
def separation_3d(
    a: AbstractVector, b: AbstractVector, /
) -> Distance | u.AbstractQuantity:
    """Straight-line distance between two points.

    The Euclidean distance between the points in their common Cartesian chart,
    invariant to the chart and component units of either operand.  Returns a
    `Distance` for unitful vectors, or a dimensionless `unxt.Quantity` for
    unitless ones (a dimensionless magnitude is not a `Distance`).

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc

    A 3-4-5 right triangle:

    >>> p = cx.Point.from_([3.0, 0.0, 0.0], "m")
    >>> q = cx.Point.from_([0.0, 4.0, 0.0], "m")
    >>> cx.separation_3d(p, q).round(2)
    Distance(5., 'm')

    Chart- and unit-invariant -- the same points expressed differently give the
    same distance:

    >>> cx.separation_3d(p, q.cconvert(cxc.sph3d)).round(2)
    Distance(5., 'm')
    >>> q_km = cx.Point.from_([0.0, 0.004, 0.0], "km")
    >>> cx.separation_3d(p, q_km).uconvert("m").round(2)
    Distance(5., 'm')

    """
    d = _cartesian_components(a, b)
    dist = jnp.sqrt(jnp.sum((d[0] - d[1]) ** 2, axis=-1))
    if u.dimension_of(dist) == _LENGTH:
        return Distance.from_(dist)  # ty: ignore[invalid-return-type]
    return dist  # ty: ignore[invalid-return-type]


@dispatch
def separation(a: AbstractVector, b: AbstractVector, /) -> Angle:
    """Angular separation between two directions.

    The angle subtended at the origin by the two points, computed in their
    common Cartesian chart with a numerically stable, dimension-agnostic
    formula (no cross product), and invariant to chart and component units.

    A vector *at* the origin has no direction, so its angular separation is
    undefined; the result is ``nan`` (rather than raising, so the function
    stays ``jit``/``vmap`` friendly).

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc

    Two orthogonal directions are 90 degrees apart:

    >>> p = cx.Point.from_([3.0, 0.0, 0.0], "m")
    >>> q = cx.Point.from_([0.0, 4.0, 0.0], "m")
    >>> cx.separation(p, q).uconvert("deg").round(2)
    Angle(90., 'deg')

    Chart-invariant:

    >>> cx.separation(p, q.cconvert(cxc.sph3d)).uconvert("deg").round(2)
    Angle(90., 'deg')

    """
    qa, qb = _cartesian_components(a, b)
    ua = qa / jnp.sqrt(jnp.sum(qa**2, axis=-1, keepdims=True))
    ub = qb / jnp.sqrt(jnp.sum(qb**2, axis=-1, keepdims=True))
    # theta = 2 * atan2(|uhat - vhat|, |uhat + vhat|): stable for all angles and
    # any dimension, unlike arccos(dot) which loses precision near 0 and pi.
    sub = jnp.sqrt(jnp.sum((ua - ub) ** 2, axis=-1))
    add = jnp.sqrt(jnp.sum((ua + ub) ** 2, axis=-1))
    return Angle.from_(2.0 * jnp.arctan2(sub, add))  # ty: ignore[invalid-return-type]
