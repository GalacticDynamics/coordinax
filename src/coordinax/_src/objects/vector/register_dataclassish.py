"""Vector."""

__all__ = ()


from typing import Any

import plum

from .base import Vector


@plum.dispatch
def replace(obj: Vector, /, **kwargs: Any) -> Vector:
    """Replace fields of a vector.

    Examples
    --------
    >>> import dataclassish
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "m")
    >>> vec
    Vector(...)

    >>> print(dataclassish.replace(vec, z=u.Q(4, "km")))
    <Vector: chart=Cart3D, role=Point (x[m], y[m], z[km])
        [1 2 4]>

    """
    chart = kwargs.pop("chart", obj.chart)
    role = kwargs.pop("role", obj.role)
    return Vector(data=obj.data | kwargs, chart=chart, role=role)
