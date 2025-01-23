"""Sequence of Operators."""

__all__ = ["Pipe", "convert_to_pipe_operators"]

import textwrap
from dataclasses import replace
from typing import Any, final

import equinox as eqx
from plum import dispatch

from .base import AbstractOperator
from .composite import AbstractCompositeOperator


@dispatch.abstract
def convert_to_pipe_operators(inp: Any, /) -> tuple[AbstractOperator, ...]:
    raise NotImplementedError  # pragma: no cover


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

    operators: tuple[AbstractOperator, ...] = eqx.field(
        converter=convert_to_pipe_operators
    )

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


# ==============================================================
# Constructor


@dispatch
def convert_to_pipe_operators(
    inp: tuple[AbstractOperator, ...] | list[AbstractOperator],
) -> tuple[AbstractOperator, ...]:
    """Convert to a tuple of operators.

    Examples
    --------
    >>> import coordinax as cx

    >>> op1 = cx.ops.GalileanRotation([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> op2 = cx.ops.Identity()
    >>> convert_to_pipe_operators((op1, op2))
    (GalileanRotation(rotation=i32[3,3]), Identity())

    """
    return tuple(inp)


@dispatch
def convert_to_pipe_operators(inp: AbstractOperator) -> tuple[AbstractOperator, ...]:
    """Convert to a tuple of operators.

    Examples
    --------
    >>> import coordinax as cx

    >>> op1 = cx.ops.GalileanRotation([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> convert_to_pipe_operators(op1)
    (GalileanRotation(rotation=i32[3,3]),)

    """
    return (inp,)


@dispatch
def convert_to_pipe_operators(inp: Pipe) -> tuple[AbstractOperator, ...]:
    """Convert to a tuple of operators.

    Examples
    --------
    >>> import coordinax as cx

    >>> op1 = cx.ops.Identity()
    >>> op2 = cx.ops.Identity()
    >>> pipe = cx.ops.Pipe((op1, op2))
    >>> convert_to_pipe_operators(pipe)
    (Identity(), Identity())

    """
    return inp.operators
