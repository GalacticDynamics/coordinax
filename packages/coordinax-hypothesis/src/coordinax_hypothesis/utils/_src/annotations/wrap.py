"""Utilities."""

__all__ = (
    "AbstractNotIntrospectable",
    "wrap_if_not_inspectable",
    "RECOGNIZE_NONINTROSPECTABLE",
)

from dataclasses import dataclass

from collections.abc import Callable
from typing import Final, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AbstractNotIntrospectable(Generic[T]):
    """Wrapper to indicate a non-introspectable annotation."""

    ann: T


RECOGNIZE_NONINTROSPECTABLE: Final[
    list[tuple[Callable[[object], bool], type[AbstractNotIntrospectable]]]
] = []


def wrap_if_not_inspectable(ann: T, /) -> T | AbstractNotIntrospectable[T]:
    """Wrap a jaxtyping annotation in a wrapper class.

    We need to special-case jaxtyping-decorated annotations, since
    <class 'jaxtyping.Shaped'> is uncheckable at runtime.

    """
    for check, wrapper in RECOGNIZE_NONINTROSPECTABLE:
        if check(ann):
            return wrapper(ann)

    return ann
