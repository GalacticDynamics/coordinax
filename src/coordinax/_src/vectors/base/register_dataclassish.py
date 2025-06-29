"""Register Support with Dataclassish."""

__all__: list[str] = []


from collections.abc import Callable
from dataclasses import Field
from typing import Any

from plum import dispatch

from dataclassish import asdict, field_items, fields

from .attribute import VectorAttribute
from .base import AbstractVectorLike
from .flags import AttrFilter


def _is_attr(obj_cls: type[AbstractVectorLike], name: str, /) -> bool:
    """Check if the attribute is a `VectorAttribute`."""
    return isinstance(getattr(obj_cls, name, None), VectorAttribute)


# -------------------------------------------------------------------


@dispatch
def fields(flag: type[AttrFilter], obj: object) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the fields of an object, filtering out `VectorAttribute`s.

    Examples
    --------
    >>> from dataclassish import fields
    >>> import coordinax.vecs as cxv

    >>> w = cxv.CartesianPos3D.from_([1, 2, 3], "m")
    >>> [f.name for f in fields(cxv.AttrFilter, w)]
    ['x', 'y', 'z']

    """
    return fields(flag, type(obj))


@dispatch
def fields(flag: type[AttrFilter], obj_cls: type[object]) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the fields of an object, filtering out `VectorAttribute`s.

    Examples
    --------
    >>> import dataclasses
    >>> import dataclassish
    >>> import coordinax.vecs as cxv

    >>> [f.name for f in dataclasses.fields(cxv.CartesianPos3D)]
    ['x', 'y', 'z']

    >>> [f.name for f in dataclassish.fields(cxv.AttrFilter, cxv.CartesianPos3D)]
    ['x', 'y', 'z']

    """
    return tuple(
        field for field in fields(obj_cls) if not _is_attr(obj_cls, field.name)
    )


# -------------------------------------------------------------------


@dispatch
def asdict(
    flag: type[AttrFilter],
    obj: AbstractVectorLike,
    /,
    *,
    dict_factory: Callable[[list[tuple[str, Any]]], dict[str, Any]] = dict,
) -> dict[str, Any]:
    """Return object, filtering out `VectorAttribute`s, fields as a dict.

    Examples
    --------
    >>> from dataclassish import asdict
    >>> import coordinax.vecs as cxv

    >>> w = cxv.CartesianPos3D.from_([1, 2, 3], "m")
    >>> asdict(cxv.AttrFilter, w)
    {'x': {'value': Array(1, dtype=int32), 'unit': Unit("m")},
     'y': {'value': Array(2, dtype=int32), 'unit': Unit("m")},
     'z': {'value': Array(3, dtype=int32), 'unit': Unit("m")}}

    """
    obj_cls = type(obj)
    return dict_factory(
        [(k, v) for k, v in asdict(obj).items() if not _is_attr(obj_cls, k)]
    )


# -------------------------------------------------------------------


@dispatch
def field_keys(flag: type[AttrFilter], obj: AbstractVectorLike, /) -> tuple[str, ...]:
    """Return field keys, filtering out `VectorAttribute`s.

    Examples
    --------
    >>> from dataclassish import field_keys
    >>> import coordinax.vecs as cxv

    >>> w = cxv.CartesianPos3D.from_([1, 2, 3], "m")
    >>> field_keys(cxv.AttrFilter, w)
    ('x', 'y', 'z')

    """
    return tuple(f.name for f in fields(flag, obj))


@dispatch
def field_keys(
    flag: type[AttrFilter], obj_cls: type[AbstractVectorLike], /
) -> tuple[str, ...]:
    """Return field keys, filtering out `VectorAttribute`s.

    Examples
    --------
    >>> from dataclassish import field_keys
    >>> import coordinax.vecs as cxv

    >>> field_keys(cxv.AttrFilter, cxv.CartesianPos3D)
    ('x', 'y', 'z')

    """
    return tuple(f.name for f in fields(flag, obj_cls))


# -------------------------------------------------------------------


@dispatch
def field_values(flag: type[AttrFilter], obj: AbstractVectorLike, /) -> tuple[Any, ...]:
    """Return field values, filtering out `VectorAttribute`s.

    Examples
    --------
    >>> from dataclassish import asdict
    >>> import coordinax.vecs as cxv

    >>> w = cxv.CartesianPos3D.from_([1, 2, 3], "m")
    >>> field_values(cxv.AttrFilter, w)
    (Quantity(Array(1, dtype=int32), unit='m'),
     Quantity(Array(2, dtype=int32), unit='m'),
     Quantity(Array(3, dtype=int32), unit='m'))

    """
    obj_cls = type(obj)
    return tuple(v for k, v in field_items(obj) if not _is_attr(obj_cls, k))


# -------------------------------------------------------------------


@dispatch
def field_items(
    flag: type[AttrFilter], obj: AbstractVectorLike, /
) -> tuple[tuple[str, Any], ...]:
    """Return field items, filtering out `VectorAttribute`s.

    Examples
    --------
    >>> from dataclassish import field_items
    >>> import coordinax as cx

    >>> w = cxv.CartesianPos3D.from_([1, 2, 3], "m")
    >>> field_items(w)
    (('x', Quantity(Array(1, dtype=int32), unit='m')),
     ('y', Quantity(Array(2, dtype=int32), unit='m')),
     ('z', Quantity(Array(3, dtype=int32), unit='m')))

    """
    obj_cls = type(obj)
    return tuple((k, v) for k, v in field_items(obj) if not _is_attr(obj_cls, k))
