"""Base classes for operators on coordinates and potentials."""

__all__ = ["simplify_op"]

from functools import singledispatch

from .base import AbstractOperator


@singledispatch
def simplify_op(op: AbstractOperator, /) -> AbstractOperator:
    """Simplify an operator.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax.operators as co

    An operator with real effect cannot be simplified:

    >>> shift = Quantity([1, 0, 0], "m")  # no shift
    >>> op = co.GalileanSpatialTranslationOperator(shift)
    >>> co.simplify_op(op)
    GalileanSpatialTranslationOperator(
      translation=CartesianPosition3D( ... )
    )

    An operator with no effect can be simplified:

    >>> shift = Quantity([0, 0, 0], "m")  # no shift
    >>> op = co.GalileanSpatialTranslationOperator(shift)
    >>> co.simplify_op(op)
    IdentityOperator()

    """
    return op
