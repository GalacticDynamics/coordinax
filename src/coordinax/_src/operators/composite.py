"""Base classes for operators on coordinates."""

__all__ = ("AbstractCompositeOperator",)

from dataclasses import replace

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, final, overload, runtime_checkable

import equinox as eqx

from dataclassish import DataclassInstance

from .base import AbstractOperator

if TYPE_CHECKING:
    import coordinax.ops  # noqa: ICN001


@final
@runtime_checkable
class HasOperatorsAttr(DataclassInstance, Protocol):  # type: ignore[misc]
    """Protocol for classes with an `operators` attribute."""

    operators: tuple[AbstractOperator, ...]


class AbstractCompositeOperator(AbstractOperator):
    """Abstract Composite Operator.

    This is the base class for all composite operations.

    See Also
    --------
    :class:`coordinax.ops.Pipe`
    :class:`coordinax.ops.GalileanOp`

    """

    operators: eqx.AbstractVar[tuple[AbstractOperator, ...]]
    """The sequence of operators in the composite operator."""

    # ===========================================
    # Operator API

    @property
    def inverse(self: "AbstractCompositeOperator") -> "coordinax.ops.Pipe":
        """The inverse of the operator.

        This is the sequence of the inverse of each operator in reverse order.

        Examples
        --------
        >>> import coordinax as cx

        >>> shift = cx.ops.GalileanOp.from_([1, 2, 3], "km")
        >>> boost = cx.ops.Add.from_([1, 2, 3], "km/s")
        >>> pipe = cx.ops.Pipe((shift, boost))
        >>> pipe.inverse
        Pipe(( Add(...), GalileanOp(...) ))

        """
        from .pipe import Pipe  # noqa: PLC0415

        return Pipe(tuple(op.inverse for op in reversed(self.operators)))

    # ===========================================
    # Python API

    @overload
    def __getitem__(self, key: int) -> AbstractOperator: ...

    @overload
    def __getitem__(self, key: slice) -> "AbstractCompositeOperator": ...

    def __getitem__(
        self, key: int | slice
    ) -> "AbstractOperator | AbstractCompositeOperator":
        """Get one or more operators from the composite operator.

        This returns either a single operator or a new composite operator,
        depending if the result of the getitem on the `operators` attribute is a
        single operator or a tuple of operators.

        """
        ops = self.operators[key]
        if isinstance(ops, AbstractOperator):
            return ops
        return replace(self, operators=ops)

    def __iter__(self: HasOperatorsAttr) -> Iterator[AbstractOperator]:
        """Iterate over the operators in the composite operator."""
        return iter(self.operators)
