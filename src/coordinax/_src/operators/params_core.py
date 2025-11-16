"""Coordinax Operator package."""

__all__ = (
    "AbstractParameter",
    "ConstantParameter",
    "CustomParameter",
)


from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, Self, TypeVar, final

T = TypeVar("T")


class AbstractParameter(metaclass=ABCMeta):
    """A Descriptor for Operator Parameters."""

    @abstractmethod
    def __call__(self, *args: Any, **kw: Any) -> Any:
        """Apply the parameter to given arguments."""
        msg = "Subclasses must implement __call__ method."
        raise NotImplementedError(msg)


@final
@dataclass(frozen=True, slots=True)
class ConstantParameter(AbstractParameter, Generic[T]):
    """A constant parameter."""

    value: T
    """The constant value of the parameter."""

    def __call__(self, _: Any = 0, /, *__: Any, **___: Any) -> T:
        """Return the constant value, ignoring any arguments."""
        return self.value

    def __neg__(self) -> Self:
        return replace(self, value=-self.value)


@final
class CustomParameter(AbstractParameter):
    """A parameter defined by a custom function."""

    func: Callable

    def __init__(self, func: Any) -> None:
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the custom function to the given arguments."""
        return self.func(*args, **kwargs)
