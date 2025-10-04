"""Base classes for operators on coordinates."""

__all__: tuple[str, ...] = ()

from typing import Any, Protocol, runtime_checkable

from dataclassish import DataclassInstance

from .base import AbstractOperator


@runtime_checkable
class HasOperatorsAttr(DataclassInstance, Protocol):  # type: ignore[misc]
    """Protocol for classes with an `operators` attribute."""

    operators: tuple[AbstractOperator, ...]


@runtime_checkable
class CanAddandNeg(Protocol):
    """Protocol for classes that support addition and negation."""

    def __add__(self, other: Any, /) -> Any: ...
    def __neg__(self, /) -> Any: ...
