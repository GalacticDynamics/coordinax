"""Base classes for operators on coordinates."""

__all__: tuple[str, ...] = ()

from typing import Protocol, runtime_checkable

from dataclassish import DataclassInstance

from .base import AbstractOperator


@runtime_checkable
class HasOperatorsAttr(DataclassInstance, Protocol):  # type: ignore[misc]
    """Protocol for classes with an `operators` attribute."""

    operators: tuple[AbstractOperator, ...]
