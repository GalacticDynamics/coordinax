"""Register coordinax-related dispatches."""

__all__: tuple[str, ...] = ()

from dataclasses import replace

from typing import Any, cast

import plum

import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .point import Point
from coordinax.internal.custom_types import CDict, OptUSys

CHART_MSMTCH = "from_chart {0} does not match the point's chart {1.chart}"

# ===================================================================
# Vector conversion


@plum.dispatch
def cconvert(
    from_vec: Point, to_chart: cxc.AbstractChart, /, *, usys: OptUSys = None
) -> Point:
    """Convert a point from one chart to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cx.cconvert(vec, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    # Call the `cconvert` function on the data from the vector's kind
    p = cxr.cconvert(from_vec.data, from_vec.chart, from_vec.rep, to_chart, usys=usys)
    # Return a new vector
    return replace(from_vec, data=p, chart=to_chart)


@plum.dispatch
def cconvert(
    from_vec: Point,
    from_chart: cxc.AbstractChart,
    to_chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Point:
    """Convert a vector from one chart to another.

    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> sph_vec = cx.cconvert(vec, cx.cart3d, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    if from_chart != from_vec.chart:
        raise ValueError(CHART_MSMTCH.format(from_chart, from_vec))

    out = cxr.cconvert(from_vec, to_chart, usys=usys)
    return cast("Point", out)


# -------------------------------------------------


@plum.dispatch
def pt_map(
    from_vec: Point, to_chart: cxc.AbstractChart, /, *, usys: OptUSys = None
) -> Point:
    """Convert a point from one chart to another.

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cxc.pt_map(vec, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    # Call `pt_map` on the data from the vector's kind
    p = cxc.pt_map(from_vec.data, from_vec.chart, from_vec.rep, to_chart, usys=usys)
    # Return a new vector
    return replace(from_vec, data=p, chart=to_chart)


@plum.dispatch
def pt_map(
    from_vec: Point,
    from_chart: cxc.AbstractChart,
    to_chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Point:
    """Convert a vector from one chart to another.

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc

    >>> vec = cx.Point.from_([1, 1, 1], "m")
    >>> sph_vec = cxc.pt_map(vec, cxc.cart3d, cx.sph3d)
    >>> print(sph_vec)
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    if from_chart != from_vec.chart:
        raise ValueError(CHART_MSMTCH.format(from_chart, from_vec))
    out = cxc.pt_map(from_vec, to_chart, usys=usys)
    return cast("Point", out)


# ===================================================================
# cdict dispatch


@plum.dispatch
def cdict(obj: Point, /) -> CDict:
    """Extract component dictionary from a Point.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import unxt as u
    >>> vec = cx.Point.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    return obj.data


# ===================================================================
# Add / Subtract


@plum.dispatch
def add(lhs: Point, rhs: Point, /) -> Point:
    """Add two points.

    For non-Cartesian charts the operation converts both operands to the
    ambient Cartesian chart, adds there, and converts the result back
    to the ``lhs`` chart.  For Cartesian charts the addition is direct.

    The result keeps the ``lhs`` chart and representation.

    Examples
    --------
    >>> import coordinax.main as cx

    >>> v1 = cx.Point.from_([1, 2, 3], "m")
    >>> v2 = cx.Point.from_([4, 5, 6], "m")
    >>> print(cxr.add(v1, v2))
    <Point: chart=Cart3D (x, y, z) [m]
        [5 7 9]>

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot add vectors with different representations: "
            f"{lhs.rep} vs {rhs.rep}."
        )
        raise TypeError(msg)

    result_data = cxr.add(lhs.data, lhs.chart, lhs.rep, rhs.data, rhs.chart, rhs.rep)
    return replace(lhs, data=result_data)


@plum.dispatch
def subtract(lhs: Point, rhs: Point, /) -> Point:
    """Subtract two vectors.

    For non-Cartesian charts the operation converts both operands to the
    ambient Cartesian chart, subtracts there, and converts the result back
    to the ``lhs`` chart.  For Cartesian charts the subtraction is direct.

    The result keeps the ``lhs`` chart and representation.

    Examples
    --------
    >>> import coordinax.main as cx

    >>> v1 = cx.Point.from_([4, 5, 6], "m")
    >>> v2 = cx.Point.from_([1, 2, 3], "m")
    >>> print(cxr.subtract(v1, v2))
    <Point: chart=Cart3D (x, y, z) [m]
        [3 3 3]>

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot subtract vectors with different representations: "
            f"{lhs.rep} vs {rhs.rep}."
        )
        raise TypeError(msg)

    result_data = cxr.subtract(
        lhs.data, lhs.chart, lhs.rep, rhs.data, rhs.chart, rhs.rep
    )
    return replace(lhs, data=result_data)


# ===================================================================
# `coordinax.representations`


@plum.dispatch
def act(op: cxfm.AbstractTransform, tau: Any, x: Point, /, **kw: Any) -> Point:
    """Act a frame transform on a Point.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cx.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "km")
    >>> vec = cx.Point.from_(q)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [km]
        [1 0 0]>

    >>> transformed_vec = cx.act(op, None, vec)
    >>> print(transformed_vec)
    <Point: chart=Cart3D (x, y, z) [km]
        [0 1 0]>

    """
    data = cxfmapi.act(op, tau, x.data, x.chart, x.rep, **kw)
    return replace(x, data=data)
