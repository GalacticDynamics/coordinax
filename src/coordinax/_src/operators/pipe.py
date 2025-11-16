"""Sequence of Operators."""

__all__ = ("Pipe",)

from dataclasses import replace
from typing import Any, final

import equinox as eqx
import wadler_lindig as wl
from plum import dispatch

from .base import AbstractOperator
from .composite import AbstractCompositeOperator
from .identity import Identity


def convert_to_operator_tuple(inp: Any, /) -> tuple[AbstractOperator, ...]:
    """Convert to a tuple of operators for `Pipe`.

    Examples
    --------
    >>> import coordinax as cx

    >>> op1 = cx.ops.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> op2 = cx.ops.Identity()
    >>> convert_to_operator_tuple((op1, op2))
    (Rotate(rotation=i32[3,3]), Identity())

    >>> op1 = cx.ops.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> convert_to_operator_tuple(op1)
    (Rotate(rotation=i32[3,3]),)

    >>> op1 = cx.ops.Identity()
    >>> op2 = cx.ops.Identity()
    >>> pipe = cx.ops.Pipe((op1, op2))
    >>> convert_to_operator_tuple(pipe)
    (Identity(), Identity())

    """
    if isinstance(inp, (tuple, list)):
        return tuple(inp)
    if isinstance(inp, AbstractOperator):
        return (inp,)
    if isinstance(inp, Pipe):
        return inp.operators

    msg = f"Cannot convert object of type {type(inp)} to a tuple of operators."
    raise ValueError(msg)


# =============================================================


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
    operators
        The sequence of operators to apply.

    Examples
    --------
    >>> import coordinax as cx

    >>> shift = cx.ops.Add.from_([1, 2, 3], "km")
    >>> boost = cx.ops.VelocityBoost.from_([1, 2, 3], "km/s")
    >>> pipe = cx.ops.Pipe((shift, boost))
    >>> pipe
    Pipe(( Add(...), VelocityBoost(...) ))

    A pipe can also be constructed by ``|``:

    >>> pipe2 = shift | boost
    >>> pipe2
    Pipe(( Add(...), VelocityBoost(...) ))

    The pipe can be simplified. For this example, we add an identity operator to
    the sequence and simplify, which will remove the identity operator.

    >>> pipe3 = pipe2 | cx.ops.Identity()
    >>> pipe3
    Pipe((
        Add(...), VelocityBoost(...), Identity()
    ))

    >>> cx.ops.simplify(pipe3)
    Pipe(( Add(...), VelocityBoost(...) ))

    Now let's call the operator on a position:

    >>> pos = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(pipe(pos))
    <CartesianPos3D: (x, y, z) [km]
        [2 4 6]>

    The pipe will also work on a position + velocity:

    >>> vel = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> print(*pipe(pos, vel), sep="\n")
    <CartesianPos3D: (x, y, z) [km]
        [2 4 6]>
    <CartesianVel3D: (x, y, z) [km / s]
        [5. 7. 9.]>

    """

    operators: tuple[AbstractOperator, ...] = eqx.field(
        converter=convert_to_operator_tuple
    )

    # ---------------------------------------------------------------

    def __or__(self, other: AbstractOperator) -> "Pipe":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> shift = cx.ops.Add.from_([1, 2, 3], "km")
        >>> boost = cx.ops.Add.from_([1, 2, 3], "km/s")
        >>> pipe = cx.ops.Pipe((shift, boost))

        >>> pipe | pipe
        Pipe(( Add(...), Add(...), Add(...), Add(...) ))

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

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation.

        This is used to generate the documentation for the operator.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.ops as cxo

        >>> shift = cxo.Add.from_([1, 2, 3], "km")
        >>> pipe = cxo.Pipe((shift, boost))
        >>> print(wl.pdoc(shift))

        >>> boost = cxo.Add.from_([1, 2, 3], "km/s")
        >>> pipe = cxo.Pipe((shift, boost))
        >>> print(wl.pdoc(pipe))

        """
        # Build docs for each operator
        docs = [wl.pdoc(op, **kwargs) for op in self.operators]
        # Wrap in ((...)) if more than one operator
        begin = wl.TextDoc("((" if len(docs) > 1 else "(")
        end = wl.TextDoc("))" if len(docs) > 1 else ")")
        # Assemble in Pipe(...)
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}") + begin,
            docs=docs,
            sep=wl.comma,
            end=end,
            indent=kwargs.get("indent", 4),
        )


# ===================================================================
# Call
# TODO: simplify dispatches


@dispatch(precedence=1)  # type: ignore[call-overload,misc]
def operate(
    self: AbstractCompositeOperator, tau: Any, /, arg: object, **kw: object
) -> object:
    """Apply the operators in a sequence."""
    for op in self.operators:
        arg = op(tau, arg, **kw)
    return arg


@dispatch(precedence=1)  # type: ignore[call-overload,misc]
def operate(
    self: AbstractCompositeOperator, tau: Any, /, *args: object, **kw: object
) -> tuple[object, ...]:
    """Apply the operators in a sequence."""
    for op in self.operators:
        args = op(tau, *args, **kw)
    return args


# ===================================================================
# Simplification


@dispatch
def simplify(op: Pipe, /) -> AbstractOperator:
    """Simplify a Pipe operator.

    Examples
    --------
    >>> import coordinax as cx

    >>> shift = cx.ops.Add.from_([1, 2, 3], "km")
    >>> identity = cx.ops.Identity()
    >>> pipe = cx.ops.Pipe((shift, identity))
    >>> pipe
    Pipe((
        Add(...), Identity()
    ))

    >>> cx.ops.simplify(pipe)
    Add(...)

    """
    # TODO: figure out how to do pairwise simplifications
    # Remove identity operators
    simplified_ops = tuple(o for o in op.operators if not isinstance(o, Identity))
    # If no operators remain, return identity
    if not simplified_ops:
        return Identity()
    # If only one operator remains, return it
    if len(simplified_ops) == 1:
        return simplified_ops[0]
    # Otherwise, return simplified pipe
    return replace(op, operators=simplified_ops)
