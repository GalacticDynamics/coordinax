"""Descriptor for Parameters."""

__all__ = ("ParameterField",)

from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast, final, overload

from dataclassish._src.converters import field

from .params_core import AbstractParameter, ConstantParameter, CustomParameter

if TYPE_CHECKING:
    import coordinax  # noqa: ICN001


T = TypeVar("T")
R = TypeVar("R")


class AbstractParameterField(Generic[T, R]):
    """Descriptor for an :class:`coordinax.ops.AbstractOperator` Parameter."""

    name: str
    """The name of the parameter."""

    # ===========================================
    # Descriptor

    def __set_name__(self, owner: type[T], name: str) -> None:
        """Set the name of the parameter."""
        object.__setattr__(self, "name", name)

    # -----------------------------
    # Getting

    @overload  # TODO: use `Self` when beartype is happy
    def __get__(self, instance: None, owner: type[T]) -> Self: ...

    @overload
    def __get__(self, instance: T, owner: None) -> R: ...

    def __get__(  # TODO: use `Self` when beartype is happy
        self,
        instance: T | None,
        owner: type[T] | None,
    ) -> Self | R:
        # Get from class
        if instance is None:
            # If the Parameter is being set as part of a dataclass constructor,
            # then we raise an AttributeError if there is no default value. This
            # is to prevent the Parameter from being set as the default value of
            # the dataclass field and erroneously included in the class'
            # ``__init__`` signature.
            if not is_dataclass(owner) or self.name not in owner.__dataclass_fields__:
                raise AttributeError

            # The normal return is the descriptor itself
            return self

        # Get from instance
        return cast("R", instance.__dict__[self.name])

    # -----------------------------

    @abstractmethod
    def __set__(self, op: T, value: R | Any) -> None:
        """Set the parameter on the operator."""
        raise NotImplementedError  # pragma: no cover


@final
@dataclass(frozen=True, slots=True)
class ParameterField(
    AbstractParameterField["coordinax.ops.AbstractOperator", AbstractParameter]
):
    """Descriptor for an :class:`coordinax.ops.AbstractOperator` Parameter."""

    name: str = field(init=False)
    """The name of the parameter."""

    def __set__(
        self, op: "coordinax.ops.AbstractOperator", value: AbstractParameter | Any
    ) -> None:
        # Convert
        if isinstance(value, AbstractParameter):
            v = value
        elif callable(value):
            v = CustomParameter(value)
        else:
            v = ConstantParameter(value)

        # Set
        op.__dict__[self.name] = v
