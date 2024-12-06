"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractCompositeOperator"]

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from dataclassish import DataclassInstance

from .base import AbstractOperator
from coordinax._src.vectors.base import AbstractPos

if TYPE_CHECKING:
    from typing import Self


@runtime_checkable
class HasOperatorsAttr(DataclassInstance, Protocol):  # type: ignore[misc]
    """Protocol for classes with an `operators` attribute."""

    operators: tuple[AbstractOperator, ...]


class AbstractCompositeOperator(AbstractOperator):
    """Abstract Composite Operator.

    This is the base class for all composite operations.

    See Also
    --------
    :class:`coordinax.operators.Sequence`
    :class:`coordinax.operators.GalileanOperator`

    """

    # ===========================================
    # Operator

    # TODO: how to have the `operators` attribute in a way that allows for both
    # writeable (in the from_) and read-only (as a property) subclasses.

    @AbstractOperator.__call__.dispatch
    def __call__(
        self: "AbstractCompositeOperator", *args: object
    ) -> tuple[object, ...]:
        """Apply the operators to the coordinates.

        This is the default implementation, which applies the operators in
        sequence, passing along the arguments.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.Sequence((cx.ops.Identity(), cx.ops.Identity()))
        >>> op(1, 2, 3)
        (1, 2, 3)

        """
        # TODO: with lax.for_i
        for op in self.operators:
            args = op(*args)
        return args

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(self: "AbstractCompositeOperator", x: AbstractPos, /) -> AbstractPos:
        """Apply the operator to the coordinates.

        This is the default implementation, which applies the operators in
        sequence.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.Sequence((cx.ops.Identity(), cx.ops.Identity()))
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> op(vec)
        CartesianPos3D( ... )

        """
        # TODO: with lax.for_i
        for op in self.operators:
            x = op(x)
        return x

    @property
    def is_inertial(self: HasOperatorsAttr) -> bool:
        """Whether the operations maintain an inertial reference frame."""
        return all(op.is_inertial for op in self.operators)

    @property
    def inverse(self: HasOperatorsAttr) -> "AbstractCompositeOperator":
        """The inverse of the operator."""
        from .sequence import Sequence

        return Sequence(tuple(op.inverse for op in reversed(self.operators)))

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
