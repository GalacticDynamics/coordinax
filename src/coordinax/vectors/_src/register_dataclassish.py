"""Vector."""

__all__ = ()


from typing import Any

import plum

from .core import Vector


@plum.dispatch
def replace(obj: Vector, /, **kwargs: Any) -> Vector:
    """Replace fields of a vector.

    Examples
    --------
    >>> import dataclassish
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "m")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    >>> print(dataclassish.replace(vec, z=u.Q(4, "km")))
    <Vector: chart=Cart3D, rep=point (x[m], y[m], z[km])
        [1 2 4]>

    """
    chart = kwargs.pop("chart", obj.chart)
    rep = kwargs.pop("rep", obj.rep)
    return Vector(data=obj.data | kwargs, chart=chart, rep=rep)
