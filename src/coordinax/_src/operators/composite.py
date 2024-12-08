"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractCompositeOperator"]

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from dataclassish import DataclassInstance

from .base import AbstractOperator

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
    :class:`coordinax.ops.Pipe`
    :class:`coordinax.ops.GalileanOperator`

    """

    # ===========================================
    # Operator

    @property
    def is_inertial(self: HasOperatorsAttr) -> bool:
        """Whether the operations maintain an inertial reference frame."""
        return all(op.is_inertial for op in self.operators)

    @property
    def inverse(self) -> "Pipe":
        """The inverse of the operator.

        This is the sequence of the inverse of each operator in reverse order.

        Examples
        --------
        >>> import coordinax as cx

        >>> shift = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
        >>> boost = cx.ops.GalileanBoost.from_([1, 2, 3], "km/s")
        >>> pipe = cx.ops.Pipe((shift, boost))
        >>> pipe.inverse
        Pipe(( GalileanBoost(...), GalileanSpatialTranslation(...) ))

        """
        from .pipe import Pipe

        return Pipe(tuple(op.inverse for op in reversed(self.operators)))

    # ===========================================
    # Pipe

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
