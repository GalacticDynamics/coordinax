"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractCompositeOperator"]

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from dataclassish import DataclassInstance

from .base import AbstractOperator
from coordinax._src.vectors.base import AbstractVector

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

    @AbstractOperator.__call__.dispatch(precedence=1)  # type: ignore[attr-defined, misc]
    def __call__(
        self: "AbstractCompositeOperator", *args: object, **kwargs: Any
    ) -> tuple[object, ...]:
        """Apply the operators to the coordinates.

        This is the default implementation, which applies the operators in
        sequence, passing along the arguments.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.Pipe((cx.ops.Identity(), cx.ops.Identity()))
        >>> op(1, 2, 3)
        (1, 2, 3)

        """
        for op in self.operators:
            args = op(*args, **kwargs)
        return args

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


# ======================================================================
# Call dispatches


@AbstractOperator.__call__.dispatch(precedence=1)  # type: ignore[attr-defined, misc]
def call(
    self: AbstractCompositeOperator, x: AbstractVector, /, **kwargs: Any
) -> AbstractVector:
    """Apply the operator to the coordinates.

    This is the default implementation, which applies the operators in
    sequence.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.Pipe((cx.ops.Identity(), cx.ops.Identity()))
    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> op(vec)
    CartesianPos3D( ... )

    """
    # TODO: with lax.for_i
    for op in self.operators:
        x = op(x, **kwargs)
    return x
