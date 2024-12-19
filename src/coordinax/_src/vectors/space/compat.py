"""Intra-ecosystem compatibility."""

__all__: list[str] = []


from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from dataclasses import Field
from typing import Any

from plum import dispatch

from .core import Space
from coordinax._src.vectors.base import AbstractVector


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def fields(obj: Space, /) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the items from a Space.

    Examples
    --------
    >>> from dataclassish import fields
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> v = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "km/s^2")

    >>> space = cx.Space(length=x, speed=v, acceleration=a)

    >>> fields(space)
    (Field(name='length',type=<class 'coordinax...CartesianPos3D'>,...),
     Field(name='speed',type=<class 'coordinax...CartesianVel3D'>,...),
     Field(name='acceleration',type=<class 'coordinax...CartesianAcc3D'>,...))

    """
    return fields.invoke(Mapping[str, Any])(obj)


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def field_keys(obj: Space, /) -> KeysView[str]:
    """Return the keys from a Space.

    Examples
    --------
    >>> from dataclassish import field_keys
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> v = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "km/s^2")

    >>> space = cx.Space(length=x, speed=v, acceleration=a)

    >>> field_keys(space)
    dict_keys(['length', 'speed', 'acceleration'])

    """
    return obj.keys()


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def field_values(obj: Space, /) -> ValuesView[AbstractVector]:
    """Return the values from a Space.

    Examples
    --------
    >>> from dataclassish import field_values
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> v = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "km/s^2")

    >>> space = cx.Space(length=x, speed=v, acceleration=a)

    >>> field_values(space)
    dict_values([CartesianPos3D(...), CartesianVel3D(...), CartesianAcc3D(...)])

    """
    return obj.values()


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def field_items(obj: Space, /) -> ItemsView[str, AbstractVector]:
    """Return the items from a Space.

    Examples
    --------
    >>> from dataclassish import field_items
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> v = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "km/s^2")

    >>> space = cx.Space(length=x, speed=v, acceleration=a)

    >>> field_items(space)
    dict_items([('length', CartesianPos3D(...)),
                ('speed', CartesianVel3D(...)),
                ('acceleration', CartesianAcc3D(...))])

    """
    return obj.items()


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `replace` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def replace(obj: Space, /, **kwargs: AbstractVector) -> Space:
    """Replace the components of the vector.

    Examples
    --------
    >>> from dataclassish import replace
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> v = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "km/s^2")

    >>> space = cx.Space(length=x, speed=v, acceleration=a)
    >>> newspace = replace(space, length=cx.CartesianPos3D.from_([3, 2, 1], "km"))
    >>> newspace["length"].x
    Quantity['length'](Array(3, dtype=int32), unit='km')

    """
    return type(obj)(**{**obj, **kwargs})
