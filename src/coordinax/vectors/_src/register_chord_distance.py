"""`chord_distance` dispatch for vector-like objects.

This registers the `~coordinax.vectors.Point` overload of
`coordinaxs.api.manifolds.chord_distance`, the counterpart to the one
`geodesic_distance` has.

Unlike that one, the operands are *not* brought into a Cartesian chart first.
A chord is measured through the ambient space a manifold is embedded in, and a
Euclidean manifold is its own ambient -- so converting to Cartesian would turn
every call into the case `chord_distance` refuses, and would raise outright for
an intrinsic sphere chart, which has no global Cartesian representation. The
second operand is mapped into the first's chart instead, and the measurement is
delegated to the manifold-level `chord_distance`.

It is *frame-strict*, as `geodesic_distance` is: coordinates in different frames
describe different physical points, so a cross-frame chord is undefined and
raises; align the operands with `to_frame` first.
"""

__all__: tuple[str, ...] = ("chord_distance",)

from typing import Any

import plum

import coordinaxs.api.manifolds as cxmapi
from .point import Point


@plum.dispatch
def chord_distance(a: Point, b: Point, /) -> Any:
    """Straight-line distance between two points, through their ambient space.

    The counterpart to `~coordinax.geodesic_distance`'s `Point` overload: that
    one measures along the manifold, this one through the space it is embedded
    in. The result is invariant to the chart each operand happens to use.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc

    A quarter turn along the equator: the arc is ``pi / 2``, the chord through
    the interior is ``sqrt(2)``.

    >>> p = cx.Point({"theta": u.Angle(jnp.pi / 2, "rad"),
    ...               "phi": u.Angle(0.0, "rad")}, chart=cxc.sph2)
    >>> q = cx.Point({"theta": u.Angle(jnp.pi / 2, "rad"),
    ...               "phi": u.Angle(jnp.pi / 2, "rad")}, chart=cxc.sph2)
    >>> round(float(cx.chord_distance(p, q)), 6)
    1.414214

    The operands need not share a chart:

    >>> r = cx.Point({"lon": u.Angle(jnp.pi / 2, "rad"),
    ...               "lat": u.Angle(0.0, "rad")}, chart=cxc.lonlat_sph2)
    >>> round(float(cx.chord_distance(p, r)), 6)
    1.414214

    Flat space is its own ambient, so it has no chord distinct from its
    geodesic:

    >>> a = cx.Point.from_([3.0, 0.0, 0.0], "m")
    >>> b = cx.Point.from_([0.0, 4.0, 0.0], "m")
    >>> try: cx.chord_distance(a, b)
    ... except NotImplementedError as e: print(str(e)[:48])
    chord_distance is a measurement through an ambie

    """
    if a.frame != b.frame:
        msg = "cannot measure chord_distance between vectors in different frames"
        raise ValueError(msg)

    if a.chart.M != b.chart.M:
        msg = "cannot measure chord_distance between vectors on different manifolds"
        raise ValueError(msg)

    # Into `a`'s chart rather than a Cartesian one: the chord is a property of
    # the manifold's embedding, and the intrinsic chart is what carries it.
    b_in_a = b.cconvert(a.chart)
    return cxmapi.chord_distance(a.chart, a.data, b_in_a.data)
