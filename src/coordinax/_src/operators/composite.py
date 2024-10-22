"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractCompositeOperator"]

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from dataclassish import DataclassInstance
from unxt import Quantity

from .base import AbstractOperator, op_call_dispatch
from coordinax._src.base import AbstractPos

if TYPE_CHECKING:
    from typing_extensions import Self


@runtime_checkable
class HasOperatorsAttr(DataclassInstance, Protocol):  # type: ignore[misc]
    """Protocol for classes with an `operators` attribute."""

    operators: tuple[AbstractOperator, ...]


class AbstractCompositeOperator(AbstractOperator):
    """Abstract Composite Operator.

    This is the base class for all composite operations.

    See Also
    --------
    :class:`coordinax.operators.OperatorSequence`
    :class:`coordinax.operators.GalileanOperator`

    """

    # ===========================================
    # Operator

    # TODO: how to have the `operators` attribute in a way that allows for both
    # writeable (in the from_) and read-only (as a property) subclasses.

    @op_call_dispatch(precedence=1)
    def __call__(self: "AbstractCompositeOperator", x: AbstractPos, /) -> AbstractPos:
        """Apply the operator to the coordinates.

        This is the default implementation, which applies the operators in
        sequence.
        """
        # TODO: with lax.scan?
        for op in self.operators:
            x = op(x)
        return x

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "AbstractCompositeOperator", x: AbstractPos, t: Quantity["time"], /
    ) -> tuple[AbstractPos, Quantity["time"]]:
        """Apply the operator to the coordinates."""
        # TODO: with lax.scan?
        for op in self.operators:
            x, t = op(x, t)
        return x, t

    @property
    def is_inertial(self: HasOperatorsAttr) -> bool:
        """Whether the operations maintain an inertial reference frame."""
        return all(op.is_inertial for op in self.operators)

    @property
    def inverse(self: HasOperatorsAttr) -> "AbstractCompositeOperator":
        """The inverse of the operator."""
        from .sequential import OperatorSequence

        return OperatorSequence(tuple(op.inverse for op in reversed(self.operators)))

    # ===========================================
    # Sequence

    @overload
    def __getitem__(self, key: int) -> AbstractOperator: ...

    @overload
    def __getitem__(self, key: slice) -> "Self": ...

    def __getitem__(self, key: int | slice) -> "AbstractOperator | Self":
        ops = self.operators[key]
        if isinstance(ops, AbstractOperator):
            return ops
        return replace(self, operators=ops)

    def __iter__(self: HasOperatorsAttr) -> Iterator[AbstractOperator]:
        return iter(self.operators)
