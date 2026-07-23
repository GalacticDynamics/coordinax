"""Separation (distance) between vectors.

:func:`separation` measures the straight-line distance between two points using
the manifold's `~coordinax.manifolds.norm`.  The *geometry* is the manifold's:
the distance is computed in the points' own (Cartesian) chart, so it is the
Euclidean distance for a flat manifold, and its dimensionality follows the
points' manifold (2-D points give a 2-D distance, 3-D points a 3-D distance).
To measure across a different-dimensional or projected space, map the points
into that space first, then call :func:`separation`.

It is *frame-strict*: coordinates in different frames describe different physical
points, so a cross-frame separation is undefined and raises; align the operands
with `to_frame` first.
"""

__all__: tuple[str, ...] = ("separation",)

from typing import Any

from plum import dispatch

import unxt as u

from .base import AbstractVector
from coordinax.distances import Distance

_LENGTH = u.dimension("length")


@dispatch
def separation(a: AbstractVector, b: AbstractVector, /) -> Distance | Any:
    """Distance between two points, via the manifold norm.

    The two points are brought into a common Cartesian chart (so the result is
    invariant to the chart and component units each operand happens to use), and
    the distance is the manifold `~coordinax.manifolds.norm` of their coordinate
    difference -- the Euclidean distance for a flat manifold.  A length result is
    returned as a `Distance`; a unitless (dimensionless) result is returned as a
    bare array.

    Dimensionality follows the points' manifold: 2-D points give a 2-D distance,
    3-D points a 3-D distance.  There is no ``separation_3d`` -- to measure in a
    particular N-D space, map the points into it first, then call `separation`.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc

    A 3-4-5 right triangle:

    >>> p = cx.Point.from_([3.0, 0.0, 0.0], "m")
    >>> q = cx.Point.from_([0.0, 4.0, 0.0], "m")
    >>> cx.separation(p, q).round(2)
    Distance(5., 'm')

    Chart- and unit-invariant -- the same points expressed differently give the
    same distance:

    >>> cx.separation(p, q.cconvert(cxc.sph3d)).round(2)
    Distance(5., 'm')
    >>> q_km = cx.Point.from_([0.0, 0.004, 0.0], "km")
    >>> cx.separation(p, q_km).uconvert("m").round(2)
    Distance(5., 'm')

    The distance lives on the points' manifold, so 2-D points give a 2-D
    distance:

    >>> p2 = cx.Point.from_([3.0, 0.0], "m")
    >>> q2 = cx.Point.from_([0.0, 4.0], "m")
    >>> cx.separation(p2, q2).round(2)
    Distance(5., 'm')

    """
    if a.frame != b.frame:
        msg = "cannot measure separation between vectors in different frames"
        raise ValueError(msg)

    ac = a.to_cartesian()
    bc = b.to_cartesian()
    if ac.chart != bc.chart:
        msg = "cannot measure separation between vectors on different manifolds"
        raise ValueError(msg)

    # Difference built per component so it works for both ``unxt.Quantity`` and
    # plain-array (unitless) leaves, then measured with the manifold's norm.
    diff = {k: bc.data[k] - ac.data[k] for k in ac.data}
    dist = ac.M.norm(diff, ac.chart, at=ac.data)

    # A length is a ``Distance``; a dimensionless magnitude is returned as-is.
    if hasattr(dist, "unit") and u.dimension_of(dist) == _LENGTH:
        return Distance.from_(dist)  # ty: ignore[invalid-return-type]
    return dist
