"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Callable
from dataclasses import dataclass, replace as _dataclass_replace
from typing import TYPE_CHECKING, Generic, TypeVar

import quaxed.array_api as xp
from dataclassish import field_values

if TYPE_CHECKING:
    from coordinax._coordinax.base import AbstractVector


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
