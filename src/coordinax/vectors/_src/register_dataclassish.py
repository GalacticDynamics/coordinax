"""dataclassish registrations for `Point` and `Coordinate`."""

__all__ = ()


import dataclasses

from collections.abc import Callable, Mapping
from typing import Any

import plum

import dataclassish

from .bundle import Coordinate
from .point import Point
from .tangent import Tangent


@plum.dispatch
def replace(obj: Point, /, **kwargs: Any) -> Point:
    """Replace fields of a point.

    >>> import dataclassish
    >>> import unxt as u
    >>> import coordinax as cx

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
    fs = {k: v for k, v in kwargs.items() if k in keys}
    component_kwargs = {k: v for k, v in kwargs.items() if k not in keys}
    # If any of the kwargs are also in the data, that's an error (ambiguous).
    if "data" in fs and component_kwargs:
        raise ValueError("Cannot pass both data and non-field kwargs.")
    # If any of the kwargs are in the data, merge them with the existing data.
    if component_kwargs:
        fs["data"] = {**obj.data, **component_kwargs}
    # Replace the fields using dataclasses.replace.
    return dataclasses.replace(obj, **fs)


# ===================================================================
# Coordinate
#
# The public fields of a `Coordinate` are its base ``point`` followed by its
# named fibres: the keywords `Coordinate(...)` and `dataclassish.replace` take.
# The storage field ``_data`` is not one of them.


def _coordinate_fields(obj: Coordinate, /) -> dict[str, Point | Tangent]:
    return {"point": obj.point, **obj._data}


@dataclassish.field_keys.dispatch  # type: ignore[attr-defined]
def field_keys(obj: Coordinate, /) -> tuple[str, ...]:
    """Return ``'point'`` followed by the fibre names.

    >>> import dataclassish
    >>> import coordinax as cx

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_([1.0, 0.0, 0.0], "m/s")
    >>> coord = cx.Coordinate(point=point, velocity=vel)
    >>> dataclassish.field_keys(coord)
    ('point', 'velocity')

    """
    return tuple(_coordinate_fields(obj))


@dataclassish.field_values.dispatch  # type: ignore[attr-defined]
def field_values(obj: Coordinate, /) -> tuple[Point | Tangent, ...]:
    """Return the base point followed by the fibres.

    >>> import dataclassish
    >>> import coordinax as cx

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_([1.0, 0.0, 0.0], "m/s")
    >>> coord = cx.Coordinate(point=point, velocity=vel)
    >>> [type(v).__name__ for v in dataclassish.field_values(coord)]
    ['Point', 'Tangent']

    """
    return tuple(_coordinate_fields(obj).values())


@dataclassish.field_items.dispatch  # type: ignore[attr-defined]
def field_items(obj: Coordinate, /) -> tuple[tuple[str, Point | Tangent], ...]:
    """Return ``(name, vector)`` pairs: the base point, then the fibres.

    They are the keyword arguments that rebuild the bundle.

    >>> import dataclassish
    >>> import coordinax as cx

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_([1.0, 0.0, 0.0], "m/s")
    >>> coord = cx.Coordinate(point=point, velocity=vel)
    >>> [k for k, _ in dataclassish.field_items(coord)]
    ['point', 'velocity']
    >>> bool(cx.Coordinate(**dict(dataclassish.field_items(coord))) == coord)
    True

    """
    return tuple(_coordinate_fields(obj).items())


@dataclassish.fields.dispatch  # type: ignore[attr-defined]
def fields(obj: Coordinate, /) -> tuple[dataclasses.Field, ...]:  # type: ignore[type-arg]
    """Return one `dataclasses.Field` per public field: the point, then the fibres.

    >>> import dataclassish
    >>> import coordinax as cx

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_([1.0, 0.0, 0.0], "m/s")
    >>> coord = cx.Coordinate(point=point, velocity=vel)
    >>> [f.name for f in dataclassish.fields(coord)]
    ['point', 'velocity']

    """
    return dataclassish.fields.invoke(Mapping[str, Any])(_coordinate_fields(obj))


@dataclassish.asdict.dispatch  # type: ignore[attr-defined]
def asdict(
    obj: Coordinate,
    /,
    *,
    dict_factory: Callable[[list[tuple[str, Any]]], dict[str, Any]] = dict,
) -> dict[str, Any]:
    """Return the public fields as a dict, keeping the vectors as vectors.

    The vectors are not themselves turned into dicts, so the result rebuilds
    the bundle.

    >>> import dataclassish
    >>> import coordinax as cx

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_([1.0, 0.0, 0.0], "m/s")
    >>> coord = cx.Coordinate(point=point, velocity=vel)
    >>> list(dataclassish.asdict(coord))
    ['point', 'velocity']
    >>> bool(cx.Coordinate(**dataclassish.asdict(coord)) == coord)
    True

    """
    return dict_factory(list(_coordinate_fields(obj).items()))
