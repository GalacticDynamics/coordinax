"""Base classes for operators on coordinates and potentials."""

__all__: list[str] = []

from plum import dispatch

from .base import AbstractOperator


@dispatch(precedence=-1)  # type: ignore[misc]  # very low priority
def simplify_op(op: AbstractOperator, /) -> AbstractOperator:
    """Return the operator unchanged.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> op = cxo.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> cxo.simplify_op(op) is op
    True

    """
    return op
