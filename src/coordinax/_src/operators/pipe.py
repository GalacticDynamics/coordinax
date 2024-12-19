"""Sequence of Operators."""

__all__ = ["Pipe"]

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

    >>> shift = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
    >>> boost = cx.ops.VelocityBoost.from_([1, 2, 3], "km/s")
    >>> pipe = cx.ops.Pipe((shift, boost))
    >>> pipe
    Pipe(( GalileanSpatialTranslation(...), VelocityBoost(...) ))

    A pipe can also be constructed by ``|``:

    >>> pipe2 = shift | boost
    >>> pipe2
    Pipe(( GalileanSpatialTranslation(...), VelocityBoost(...) ))

    The pipe can be simplified. For this example, we add an identity operator to
    the sequence and simplify, which will remove the identity operator.

    >>> pipe3 = pipe2 | cx.ops.Identity()
    >>> pipe3
    Pipe((
        GalileanSpatialTranslation(...), VelocityBoost(...), Identity()
    ))

    >>> cx.ops.simplify_op(pipe3)
    Pipe(( GalileanSpatialTranslation(...), VelocityBoost(...) ))

    Now let's call the operator on a position:

    >>> pos = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(pipe(pos))
    <CartesianPos3D (x[km], y[km], z[km])
        [2 4 6]>

    The pipe will also work on a position + velocity:

    >>> vel = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> print(*pipe(pos, vel), sep="\n")
    <CartesianPos3D (x[km], y[km], z[km])
        [2 4 6]>
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [5. 7. 9.]>

    """

    operators: tuple[AbstractOperator, ...] = eqx.field(converter=_converter_seq)

    # ---------------------------------------------------------------

    def __or__(self, other: AbstractOperator) -> "Pipe":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> shift = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
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

    def __ror__(self, other: AbstractOperator) -> "Pipe":
        """Compose with another operator."""
        # Append single operators
        return replace(self, operators=(other, *self))

    def __repr__(self) -> str:
        ops = repr(self.operators)
        if "\n" in ops:
            ops = "(\n" + textwrap.indent(ops[1:-1], "    ") + "\n)"
        return f"{self.__class__.__name__}({ops})"
