"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import array_api_jax_compat as xp

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from coordinax._base import AbstractVectorBase


def dataclass_values(obj: "DataclassInstance") -> Iterator[Any]:
    """Return the values of a dataclass instance."""
    yield from (getattr(obj, f.name) for f in fields(obj))


def dataclass_items(obj: "DataclassInstance") -> Iterator[tuple[str, Any]]:
    """Return the field names and values of a dataclass instance."""
    yield from ((f.name, getattr(obj, f.name)) for f in fields(obj))


def full_shaped(obj: "AbstractVectorBase", /) -> "AbstractVectorBase":
    """Return the vector, fully broadcasting all components."""
    arrays = xp.broadcast_arrays(*dataclass_values(obj))
    return replace(obj, **dict(zip(obj.components, arrays, strict=True)))


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
