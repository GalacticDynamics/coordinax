"""Point."""

__all__ = ()


import dataclasses

from typing import Any

import plum

from .point import Point


@plum.dispatch
def replace(obj: Point, /, **kwargs: Any) -> Point:
    """Replace fields of a point.

    Examples
    --------
    >>> import dataclassish
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Point.from_([1, 2, 3], "m")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> print(dataclassish.replace(vec, z=u.Q(4, "km")))
    <Point: chart=Cart3D (x[m], y[m], z[km])
        [1 2 4]>

    """
    # Get all the kwarg fields that are part of Point, reserving the rest to
    # pass to the constructor.
    keys = obj.__dataclass_fields__.keys()
    fs = {k: kwargs.pop(k) for k in kwargs if k in keys}
    # If any of the kwargs are also in the data, that's an error (ambiguous).
    if "data" in fs and kwargs:
        raise ValueError("Cannot pass both data and non-field kwargs.")
    # If any of the kwargs are in the data, merge them with the existing data.
    if kwargs:
        fs["data"] = {**obj.data, **kwargs}
    # Replace the fields using dataclasses.replace.
    return dataclasses.replace(obj, **fs)
