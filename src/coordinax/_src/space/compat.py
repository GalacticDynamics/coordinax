"""Space."""

__all__: list[str] = []


from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from dataclasses import Field
from typing import Any

from plum import dispatch

from .core import Space
from coordinax._src.base import AbstractVector


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def fields(obj: Space, /) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the items from a Space."""
    return fields.invoke(Mapping[str, Any])(obj)


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def field_keys(obj: Space, /) -> KeysView[str]:
    """Return the keys from a Space."""
    return obj.keys()


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def field_values(obj: Space, /) -> ValuesView[AbstractVector]:
    """Return the values from a Space."""
    return obj.values()


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def field_items(obj: Space, /) -> ItemsView[str, AbstractVector]:
    """Return the items from a Space."""
    return obj.items()


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `replace` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def replace(obj: Space, /, **kwargs: AbstractVector) -> Space:
    """Replace the components of the vector."""
    return type(obj)(**{**obj, **kwargs})
