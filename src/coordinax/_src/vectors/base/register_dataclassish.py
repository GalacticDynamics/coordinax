"""Register Support with Dataclassish."""

__all__: list[str] = []


from collections.abc import Callable
from dataclasses import Field
from typing import Any

from plum import dispatch

from dataclassish import asdict, field_items, fields

from .attribute import VectorAttribute
from .flags import AttrFilter
from .vector import AbstractVector


def _is_attr(obj_cls: type[AbstractVector], name: str) -> bool:
    """Check if the attribute is a `VectorAttribute`."""
    return isinstance(getattr(obj_cls, name, None), VectorAttribute)


# -------------------------------------------------------------------


@dispatch
def fields(flag: type[AttrFilter], obj: object) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the fields of an object, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import fields
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> [f.name for f in fields(cx.vecs.AttrFilter, w)]
    ['t', 'q']

    """
    return fields(flag, type(obj))


@dispatch
def fields(flag: type[AttrFilter], obj_cls: type[object]) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the fields of an object, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> import dataclasses
    >>> import dataclassish
    >>> import coordinax as cx

    >>> [f.name for f in dataclasses.fields(cx.FourVector)]
    ['t', 'q', 'c']

    >>> [f.name for f in dataclassish.fields(cx.vecs.AttrFilter, cx.FourVector)]
    ['t', 'q']

    """
    return tuple(
        field for field in fields(obj_cls) if not _is_attr(obj_cls, field.name)
    )


# -------------------------------------------------------------------


@dispatch
def asdict(
    flag: type[AttrFilter],
    obj: AbstractVector,
    /,
    *,
    dict_factory: Callable[[list[tuple[str, Any]]], dict[str, Any]] = dict,
) -> dict[str, Any]:
    """Return object, filtering out `VectorAttribute`s, fields as a dict.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import asdict
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> asdict(cx.vecs.AttrFilter, w).keys()
    dict_keys(['t', 'q'])

    """
    obj_cls = type(obj)
    return dict_factory(
        [(k, v) for k, v in asdict(obj).items() if not _is_attr(obj_cls, k)]
    )


# -------------------------------------------------------------------


@dispatch
def field_keys(flag: type[AttrFilter], obj: AbstractVector, /) -> tuple[str, ...]:
    """Return field keys, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import field_keys
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> field_keys(cx.vecs.AttrFilter, w)
    ('t', 'q')

    """
    return tuple(f.name for f in fields(flag, obj))


@dispatch
def field_keys(
    flag: type[AttrFilter], obj_cls: type[AbstractVector], /
) -> tuple[str, ...]:
    """Return field keys, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import field_keys
    >>> import coordinax as cx

    >>> field_keys(cx.vecs.AttrFilter, cx.FourVector)
    ('t', 'q')

    """
    return tuple(f.name for f in fields(flag, obj_cls))


# -------------------------------------------------------------------


@dispatch
def field_values(flag: type[AttrFilter], obj: AbstractVector, /) -> tuple[Any, ...]:
    """Return field values, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import asdict
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> field_keys(cx.vecs.AttrFilter, w)
    ('t', 'q')

    """
    obj_cls = type(obj)
    return tuple(v for k, v in field_items(obj) if not _is_attr(obj_cls, k))


# -------------------------------------------------------------------


@dispatch
def field_items(
    flag: type[AttrFilter], obj: AbstractVector, /
) -> tuple[tuple[str, Any], ...]:
    """Return field items, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import field_items
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> [(f[0], type(f[1]).__name__) for f in field_items(cx.vecs.AttrFilter, w)]
    [('t', "Quantity[PhysicalType('time')]"), ('q', 'CartesianPos3D')]

    """
    obj_cls = type(obj)
    return tuple((k, v) for k, v in field_items(obj) if not _is_attr(obj_cls, k))
