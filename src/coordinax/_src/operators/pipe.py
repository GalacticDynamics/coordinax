"""Sequence of Operators."""

__all__ = ["Pipe"]

import textwrap
from dataclasses import replace
from typing import Any, final

import equinox as eqx

from .base import AbstractOperator
from .composite import AbstractCompositeOperator
from coordinax._src.vectors.base import AbstractPos


def _converter_seq(inp: Any) -> tuple[AbstractOperator, ...]:
    if isinstance(inp, tuple):
        return inp
    if isinstance(inp, list):
        return tuple(inp)
    if isinstance(inp, Pipe):
        return inp.operators
    if isinstance(inp, AbstractOperator):
        return (inp,)

    raise TypeError


@final
class Pipe(AbstractCompositeOperator):
    r"""Operator Pipe.

    Piping refers to a process in which the output of one operation is directly
    passed as the input to another. This is a composite operator that represents
    a sequence of operations to be applied in order.

    `Pipe` operators can be created using the 'pipe' syntax `op1 | op2`. A Pipe
    operator created as ``FG = F | G``, when evaluated, is equivalent to
    evaluating $g \circ f = g(f(x))$. Note the order of the operators!

    :::{note}

    The `|` operator works differently from the functional composition operator
    $\circ$, which is sadly not supported in Python. The `|` operator is like
    the Unix Shell pipe operator, where output is passed left-to-right. This
    order can be seen in the indexing of the operators in the `Pipe` object.

    :::

    Parameters
    ----------
    operators : tuple[AbstractOperator, ...]
        The sequence of operators to apply.

    Examples
    --------
    >>> import coordinax as cx

    >>> shift = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "kpc")
    >>> boost = cx.ops.GalileanBoost.from_([1, 2, 3], "km/s")
    >>> pipe = cx.ops.Pipe((shift, boost))
    >>> pipe
    Pipe(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    A sequence of operators can also be constructed by ``|``:

    >>> pipe2 = shift | boost
    >>> pipe2
    Pipe(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    The sequence of operators can be simplified. For this example, we add an
    identity operator to the sequence:

    >>> pipe3 = pipe2 | cx.ops.Identity()
    >>> pipe3
    Pipe((
        GalileanSpatialTranslation(...), GalileanBoost(...), Identity()
    ))

    >>> cx.ops.simplify_op(pipe3)
    Pipe(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    """

    operators: tuple[AbstractOperator, ...] = eqx.field(converter=_converter_seq)

    # ---------------------------------------------------------------

    @AbstractOperator.__call__.dispatch
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

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(
        self: "AbstractCompositeOperator", x: AbstractPos, /, **kwargs: Any
    ) -> AbstractPos:
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

    # ---------------------------------------------------------------

    def __or__(self, other: AbstractOperator) -> "Pipe":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> shift = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "kpc")
        >>> boost = cx.ops.GalileanBoost.from_([1, 2, 3], "km/s")
        >>> pipe = cx.ops.Pipe((shift, boost))

        >>> pipe | pipe
        Pipe(( GalileanSpatialTranslation(...), GalileanBoost(...),
               GalileanSpatialTranslation(...), GalileanBoost(...) ))

        """
        # Concatenate sequences
        if isinstance(other, type(self)):
            return replace(self, operators=self.operators + other.operators)
        # Append single operators
        return replace(self, operators=(*self, other))

    def __repr__(self) -> str:
        ops = repr(self.operators)
        if "\n" in ops:
            ops = "(\n" + textwrap.indent(ops[1:-1], "    ") + "\n)"
        return f"{self.__class__.__name__}({ops})"
