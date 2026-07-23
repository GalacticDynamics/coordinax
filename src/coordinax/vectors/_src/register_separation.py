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

from jaxtyping import Array
from typing import Any

from plum import dispatch

import quaxed.numpy as jnp

from .base import AbstractVector
from coordinax.angles import Angle
from coordinax.distances import Distance


def _cartesian_components(
    a: AbstractVector, b: AbstractVector
) -> tuple[Array, Array, Any]:
    """Return ``a`` and ``b`` as stacked Cartesian arrays in a shared unit.

    Guards that the two vectors share a frame and a Cartesian chart, then strips
    every component to the first operand's unit so the results are plain arrays
    ready for elementwise arithmetic.
    """
    if a.frame != b.frame:
        msg = "cannot measure separation between vectors in different frames"
        raise ValueError(msg)

    ac = a.to_cartesian()
    bc = b.to_cartesian()
    if ac.chart != bc.chart:
        msg = "cannot measure separation between vectors on different manifolds"
        raise ValueError(msg)

    unit = next(iter(ac.data.values())).unit
    ca = jnp.stack([ac.data[k].ustrip(unit) for k in ac.data])
    cb = jnp.stack([bc.data[k].ustrip(unit) for k in ac.data])
    return ca, cb, unit


@dispatch
def separation_3d(a: AbstractVector, b: AbstractVector, /) -> Distance:
    """Straight-line distance between two points.

    The Euclidean distance between the points in their common Cartesian chart,
    invariant to the chart and component units of either operand.

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
    ca, cb, unit = _cartesian_components(a, b)
    d = ca - cb
    return Distance(jnp.sqrt(jnp.sum(d**2, axis=0)), unit)


@dispatch
def separation(a: AbstractVector, b: AbstractVector, /) -> Angle:
    """Angular separation between two directions.

    The angle subtended at the origin by the two points, computed in their
    common Cartesian chart with a numerically stable, dimension-agnostic
    formula (no cross product), and invariant to chart and component units.

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
    ca, cb, _ = _cartesian_components(a, b)
    ua = ca / jnp.sqrt(jnp.sum(ca**2, axis=0))
    ub = cb / jnp.sqrt(jnp.sum(cb**2, axis=0))
    # theta = 2 * atan2(|uhat - vhat|, |uhat + vhat|): stable for all angles and
    # any dimension, unlike arccos(dot) which loses precision near 0 and pi.
    sub = jnp.sqrt(jnp.sum((ua - ub) ** 2, axis=0))
    add = jnp.sqrt(jnp.sum((ua + ub) ** 2, axis=0))
    return Angle(2.0 * jnp.arctan2(sub, add), "rad")
