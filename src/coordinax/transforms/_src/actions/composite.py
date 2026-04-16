"""Base classes for operators on coordinates."""

__all__ = ("AbstractCompositeTransform",)

from dataclasses import replace

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, final, overload, runtime_checkable

import equinox as eqx

from dataclassish import DataclassInstance

from .base import AbstractTransform

if TYPE_CHECKING:
    import coordinax.transforms  # noqa: ICN001


@final
@runtime_checkable
class HasTransformsAttr(DataclassInstance, Protocol):
    """Protocol for classes with a `transforms` attribute."""

    transforms: tuple[AbstractTransform, ...]


class AbstractCompositeTransform(AbstractTransform):
    """Abstract Composite Operator.

    This is the base class for all composite operations.

    See Also
    --------
    {class}`coordinax.transforms.Composed`
    GalileanOp

    """

    transforms: eqx.AbstractVar[tuple[AbstractTransform, ...]]
    """The sequence of operators in the composite operator."""

    # ===========================================
    # Operator API

    @property
    def inverse(self: "AbstractCompositeTransform") -> "coordinax.transforms.Composed":
        """The inverse of the operator.

        This is the sequence of the inverse of each operator in reverse order.

        Examples
        --------
        >>> import coordinax.transforms as cxfm
        >>> import unxt as u

        >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
        >>> rotate = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        >>> pipe = cxfm.Composed((shift, rotate))
        >>> pipe.inverse
        Composed((...))

        """
        from .composed import Composed  # noqa: PLC0415

        return Composed(tuple(op.inverse for op in reversed(self.transforms)))

    # ===========================================
    # Python API

    @overload
    def __getitem__(self, key: int) -> AbstractTransform: ...

    @overload
    def __getitem__(self, key: slice) -> "AbstractCompositeTransform": ...

    def __getitem__(
        self, key: int | slice
    ) -> "AbstractTransform | AbstractCompositeTransform":
        """Get one or more transform from the composite operator.

        This returns either a single operator or a new composite operator,
        depending if the result of the getitem on the `transforms` attribute is
        a single operator or a tuple of transform.

        """
        transforms = self.transforms[key]
        if isinstance(transforms, AbstractTransform):
            return transforms
        return replace(self, transforms=transforms)

    def __iter__(self: HasTransformsAttr) -> Iterator[AbstractTransform]:
        """Iterate over the transforms in the composite operator."""
        return iter(self.transforms)
