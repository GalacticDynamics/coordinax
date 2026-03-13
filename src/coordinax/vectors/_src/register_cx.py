"""Register coordinax-related dispatches."""

__all__: tuple[str, ...] = ()


import plum

import coordinax.charts as cxc
import coordinax.representations as cxr
from .core import Vector
from coordinax.internal.custom_types import CDict

# ===================================================================
# Vector conversion


@plum.dispatch
def vconvert(to_chart: cxc.AbstractChart, from_vec: Vector, /) -> Vector:  # type: ignore[type-arg]
    """Convert a vector from one chart to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Vector.from_([1, 1, 1], "m")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cx.vconvert(cx.sph3d, vec)
    >>> print(sph_vec)
    <Vector: chart=Spherical3D, rep=point (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    # Call the `vconvert` function on the data from the vector's kind
    p = cxr.vconvert(to_chart, from_vec.chart, from_vec.rep, from_vec.data)
    # Return a new vector
    return Vector(data=p, chart=to_chart, rep=from_vec.rep)


@plum.dispatch
def point_realization_map(to_chart: cxc.AbstractChart, from_vec: Vector, /) -> Vector:
    """Convert a vector from one chart to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Vector.from_([1, 1, 1], "m")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cx.point_realization_map(cx.sph3d, vec)
    >>> print(sph_vec)
    <Vector: chart=Spherical3D, rep=point (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    # Call `point_realization_map` on the data from the vector's kind
    p = cxc.point_realization_map(to_chart, from_vec.chart, from_vec.rep, from_vec.data)
    # Return a new vector
    return Vector(data=p, chart=to_chart, rep=from_vec.rep)


@plum.dispatch
def point_transition_map(to_chart: cxc.AbstractChart, from_vec: Vector, /) -> Vector:
    """Convert a vector from one chart to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc

    >>> vec = cx.Vector.from_([1, 1, 1], "m")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cxc.point_transition_map(cx.sph3d, vec)
    >>> print(sph_vec)
    <Vector: chart=Spherical3D, rep=point (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    # Call `point_transition_map` on the data from the vector's kind
    p = cxc.point_transition_map(to_chart, from_vec.chart, from_vec.rep, from_vec.data)
    # Return a new vector
    return Vector(data=p, chart=to_chart, rep=from_vec.rep)


# ===================================================================
# cdict dispatch


@plum.dispatch
def cdict(obj: Vector, /) -> CDict:
    """Extract component dictionary from a Vector.

    Parameters
    ----------
    obj
        A Vector object

    Returns
    -------
    dict[str, Any]
        The component dictionary from the vector's data field.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import unxt as u
    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    return obj.data
