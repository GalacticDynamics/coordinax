"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields, replace as _dataclass_replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from plum import dispatch

import quaxed.array_api as xp

if TYPE_CHECKING:
    from coordinax._base import AbstractVector


################################################################################


@runtime_checkable
class DataclassInstance(Protocol):
    """Protocol for dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, Any]]

    # B/c of https://github.com/python/mypy/issues/3939 just having
    # `__dataclass_fields__` is insufficient for `issubclass` checks.
    @classmethod
    def __subclasshook__(cls: type, c: type) -> bool:
        """Customize the subclass check."""
        return hasattr(c, "__dataclass_fields__")


@dispatch  # type: ignore[misc]
def replace(obj: DataclassInstance, /, **kwargs: Any) -> DataclassInstance:
    """Replace the fields of a dataclass instance."""
    return _dataclass_replace(obj, **kwargs)


@dispatch  # type: ignore[misc]
def field_values(obj: DataclassInstance) -> Iterator[Any]:
    """Return the values of a dataclass instance."""
    yield from (getattr(obj, f.name) for f in fields(obj))


@dispatch  # type: ignore[misc]
def field_items(obj: DataclassInstance) -> Iterator[tuple[str, Any]]:
    """Return the field names and values of a dataclass instance."""
    yield from ((f.name, getattr(obj, f.name)) for f in fields(obj))


def full_shaped(obj: "AbstractVector", /) -> "AbstractVector":
    """Return the vector, fully broadcasting all components."""
    arrays = xp.broadcast_arrays(*field_values(obj))
    return _dataclass_replace(obj, **dict(zip(obj.components, arrays, strict=True)))


################################################################################

GetterRetT = TypeVar("GetterRetT")
EnclosingT = TypeVar("EnclosingT")


@dataclass(frozen=True)
class ClassPropertyDescriptor(Generic[EnclosingT, GetterRetT]):
    """Descriptor for class properties."""

    fget: classmethod | staticmethod  # type: ignore[type-arg]

    def __get__(
        self, obj: EnclosingT | None, klass: type[EnclosingT] | None
    ) -> GetterRetT:
        if obj is None and klass is None:
            msg = "Descriptor must be accessed from an instance or class."
            raise TypeError(msg)
        if klass is None:
            assert obj is not None  # just for mypy # noqa: S101
            klass = type(obj)

        return self.fget.__get__(obj, klass)()


def classproperty(
    func: Callable[[type[EnclosingT]], GetterRetT] | classmethod | staticmethod,  # type: ignore[type-arg]
) -> ClassPropertyDescriptor[EnclosingT, GetterRetT]:
    return ClassPropertyDescriptor[EnclosingT, GetterRetT](
        fget=func if isinstance(func, classmethod | staticmethod) else classmethod(func)
    )
