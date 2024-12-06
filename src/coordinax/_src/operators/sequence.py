"""Sequence of Operators."""

__all__ = ["Sequence"]

import textwrap
from dataclasses import replace
from typing import Any, final

import equinox as eqx

from .base import AbstractOperator
from .composite import AbstractCompositeOperator


def _converter_seq(inp: Any) -> tuple[AbstractOperator, ...]:
    if isinstance(inp, tuple):
        return inp
    if isinstance(inp, list):
        return tuple(inp)
    if isinstance(inp, Sequence):
        return inp.operators
    if isinstance(inp, AbstractOperator):
        return (inp,)

    raise TypeError


@final
class Sequence(AbstractCompositeOperator):
    """Sequence of operations.

    This is the composite operator that represents a sequence of operations to
    be applied in order.

    Parameters
    ----------
    operators : tuple[AbstractOperator, ...]
        The sequence of operators to apply.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.operators as co

    >>> shift = co.GalileanSpatialTranslation(u.Quantity([1, 2, 3], "kpc"))
    >>> boost = co.GalileanBoost(u.Quantity([10, 20, 30], "km/s"))
    >>> seq = co.Sequence((shift, boost))
    >>> seq
    Sequence(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    A sequence of operators can also be constructed by ``|``:

    >>> seq2 = shift | boost
    >>> seq2
    Sequence(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    The sequence of operators can be simplified. For this example, we
    add an identity operator to the sequence:

    >>> seq3 = seq2 | co.Identity()
    >>> seq3
    Sequence((
        GalileanSpatialTranslation(...), GalileanBoost(...), Identity()
    ))

    >>> co.simplify_op(seq3)
    Sequence(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    """

    operators: tuple[AbstractOperator, ...] = eqx.field(converter=_converter_seq)

    def __or__(self, other: AbstractOperator) -> "Sequence":
        """Compose with another operator."""
        # Concatenate sequences
        if isinstance(other, type(self)):
            return replace(self, operators=self.operators + other.operators)
        # Append single operators
        return replace(self, operators=(*self, other))

    def __ror__(self, other: AbstractOperator) -> "Sequence":
        return replace(self, operators=(other, *self))

    def __repr__(self) -> str:
        ops = repr(self.operators)
        if "\n" in ops:
            ops = "(\n" + textwrap.indent(ops[1:-1], "    ") + "\n)"
        return f"{self.__class__.__name__}({ops})"
