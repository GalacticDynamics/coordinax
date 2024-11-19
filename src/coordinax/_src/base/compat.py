"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from collections.abc import Callable
from dataclasses import Field
from typing import Any

from jaxtyping import Shaped
from plum import conversion_method, convert, dispatch

import quaxed.numpy as jnp
from dataclassish import field_values
from unxt.quantity import AbstractQuantity, Quantity, UncheckedQuantity

from .attribute import VectorAttribute
from .base import AbstractVector
from .base_pos import AbstractPos
from .flags import AttrFilter
from coordinax._src.distance import Distance
from coordinax._src.utils import full_shaped

#####################################################################
# Convert to Quantity


@conversion_method(type_from=AbstractPos, type_to=AbstractQuantity)  # type: ignore[misc]
def convert_pos_to_absquantity(obj: AbstractPos, /) -> AbstractQuantity:
    """`coordinax.AbstractPos` -> `unxt.AbstractQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    cart = full_shaped(obj.represent_as(obj._cartesian_cls))  # noqa: SLF001
    return jnp.stack(tuple(field_values(cart)), axis=-1)


@conversion_method(type_from=AbstractPos, type_to=Quantity)  # type: ignore[misc]
def convert_pos_to_q(obj: AbstractPos, /) -> Quantity["length"]:
    """`coordinax.AbstractPos` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Quantity)


@conversion_method(type_from=AbstractPos, type_to=UncheckedQuantity)  # type: ignore[misc]
def convert_pos_to_uncheckedq(
    obj: AbstractPos, /
) -> Shaped[UncheckedQuantity, "*batch dims"]:
    """`coordinax.AbstractPos` -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity, UncheckedQuantity

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), UncheckedQuantity)


@conversion_method(type_from=AbstractPos, type_to=Distance)  # type: ignore[misc]
def convert_pos_to_distance(obj: AbstractPos, /) -> Shaped[Distance, "*batch dims"]:
    """`coordinax.AbstractPos` -> `coordinax.Distance`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity
    >>> from coordinax.distance import Distance

    >>> pos = cx.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.RadialPos.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.PolarPos(Quantity(1.0, "km"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0.], dtype=float32), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([0., 0., 1.], dtype=float32), unit='km')

    >>> pos = cx.CylindricalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0., 0.], dtype=float32), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Distance)


#####################################################################
# Compatibility with dataclassish


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
    >>> from coordinax import FourVector, AttrFilter

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> [f.name for f in fields(AttrFilter, w)]
    ['t', 'q']

    """
    obj_cls = type(obj)
    return tuple(field for field in fields(obj) if not _is_attr(obj_cls, field.name))


@dispatch
def fields(flag: type[AttrFilter], obj_cls: type[object]) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the fields of an object, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import fields
    >>> from coordinax import FourVector, AttrFilter

    >>> [f.name for f in fields(AttrFilter, cx.FourVector)]
    ['t', 'q']

    """
    return tuple(
        field for field in fields(obj_cls) if not _is_attr(obj_cls, field.name)
    )


# -------------------------------------------------------------------


@dispatch  # type: ignore[misc]
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
    >>> from coordinax import FourVector, AttrFilter

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> asdict(AttrFilter, w).keys()
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
    >>> from coordinax import FourVector, AttrFilter

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> field_keys(AttrFilter, w)
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
    >>> from coordinax import FourVector, AttrFilter

    >>> field_keys(AttrFilter, cx.FourVector)
    ('t', 'q')

    """
    return tuple(f.name for f in fields(flag, obj_cls))


# -------------------------------------------------------------------


@dispatch  # type: ignore[misc]
def field_values(flag: type[AttrFilter], obj: AbstractVector, /) -> tuple[Any, ...]:
    """Return field values, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import asdict
    >>> import unxt as u
    >>> from coordinax import FourVector, AttrFilter

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> field_keys(AttrFilter, w)
    ('t', 'q')

    """
    obj_cls = type(obj)
    return tuple(v for k, v in field_items(obj) if not _is_attr(obj_cls, k))


# -------------------------------------------------------------------


@dispatch  # type: ignore[misc]
def field_items(
    flag: type[AttrFilter], obj: AbstractVector, /
) -> tuple[tuple[str, Any], ...]:
    """Return field items, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import field_items
    >>> import unxt as u
    >>> from coordinax import FourVector, AttrFilter

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> [(f[0], type(f[1]).__name__) for f in field_items(AttrFilter, w)]
    [('t', "Quantity[PhysicalType('time')]"), ('q', 'CartesianPos3D')]

    """
    obj_cls = type(obj)
    return tuple((k, v) for k, v in field_items(obj) if not _is_attr(obj_cls, k))
