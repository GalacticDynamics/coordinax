"""Sequence of Operators."""

__all__ = ("Pipe",)

from dataclasses import replace

from typing import Any, final

import equinox as eqx
import plum
import wadler_lindig as wl  # type: ignore[import-untyped]

import unxt as u

from .base import AbstractOperator
from .composite import AbstractCompositeOperator
from .identity import Identity
from coordinax._src import api, charts as cxc, roles as cxr
from coordinax._src.custom_types import CsDict


def convert_to_operator_tuple(inp: Any, /) -> tuple[AbstractOperator, ...]:
    """Convert to a tuple of operators for `Pipe`.

    Examples
    --------
    >>> import coordinax as cx

    >>> op1 = cx.ops.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> op2 = cx.ops.Identity()
    >>> convert_to_operator_tuple((op1, op2))
    (Rotate(i64[3,3](jax)), Identity())

    >>> op1 = cx.ops.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> convert_to_operator_tuple(op1)
    (Rotate(i64[3,3](jax)),)

    >>> op1 = cx.ops.Identity()
    >>> op2 = cx.ops.Identity()
    >>> pipe = cx.ops.Pipe((op1, op2))
    >>> convert_to_operator_tuple(pipe)
    (Identity(), Identity())

    """
    if isinstance(inp, (tuple, list)):
        return tuple(inp)
    if isinstance(inp, Pipe):
        return inp.operators
    if isinstance(inp, AbstractOperator):
        return (inp,)

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

    >>> shift = cx.ops.Translate.from_([1, 2, 3], "km")
    >>> boost = cx.ops.Boost.from_([1, 2, 3], "km/s")
    >>> pipe = cx.ops.Pipe((shift, boost))
    >>> pipe
    Pipe(( Translate(...), Boost(...) ))

    A pipe can also be constructed by ``|``:

    >>> pipe2 = shift | boost
    >>> pipe2
    Pipe(( Translate(...), Boost(...) ))

    The pipe can be simplified. For this example, we add an identity operator to
    the sequence and simplify, which will remove the identity operator.

    >>> pipe3 = pipe2 | cx.ops.Identity()
    >>> pipe3
    Pipe(( Translate(...), Boost(...), Identity() ))

    >>> cx.ops.simplify(pipe3)
    Pipe(( Translate(...), Boost(...) ))

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

        >>> shift = cx.ops.Translate.from_([1, 2, 3], "km")
        >>> boost = cx.ops.Boost.from_([1, 2, 3], "km/s")
        >>> pipe = cx.ops.Pipe((shift, boost))

        >>> pipe | pipe
        Pipe(( Translate(...), Boost(...), Translate(...), Boost(...) ))

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

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation.

        This is used to generate the documentation for the operator.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.ops as cxo

        >>> shift = cxo.Translate.from_([1, 2, 3], "km")
        >>> boost = cxo.Boost.from_([1, 2, 3], "km/s")
        >>> pipe = cxo.Pipe((shift, boost))
        >>> wl.pprint(pipe)
        Pipe(( Translate(...), Boost(...) ))

        """
        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_name", True)
        kw.setdefault("named_unit", False)
        kw.setdefault("include_params", False)

        # Build docs for each operator
        docs = [wl.pdoc(op, **kw) for op in self.operators]
        # Wrap in ((...)) if more than one operator
        begin = wl.TextDoc("((" if len(docs) > 1 else "(")
        end = wl.TextDoc("))" if len(docs) > 1 else ")")
        # Assemble in Pipe(...)
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}") + begin,
            docs=docs,
            sep=wl.comma,
            end=end,
            indent=kw.get("indent", 4),
        )


# ===================================================================
# apply_op for Pipe


@plum.dispatch
def apply_op(
    op: Pipe,
    tau: Any,
    role: cxr.AbstractRole,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    **kw: object,
) -> CsDict:
    """Apply Pipe to a CsDict by sequentially applying each operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo
    >>> import coordinax.roles as cxr

    >>> shift = cxo.Translate.from_([1, 2, 3], "km")
    >>> pipe = cxo.Pipe((shift,))
    >>> data = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxo.apply_op(pipe, None, data, role=cxr.Point)
    {'x': Quantity..., 'y': Quantity..., 'z': Quantity...}

    """
    result = x
    for sub_op in op.operators:
        result = api.apply_op(sub_op, tau, role, chart, result, **kw)
    return result


@plum.dispatch
def apply_op(
    op: Pipe, tau: Any, x: u.AbstractQuantity, /, **kw: object
) -> u.AbstractQuantity:
    """Apply Pipe to a Quantity by sequentially applying each operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> shift = cxo.Translate.from_([1, 2, 3], "km")
    >>> pipe = cxo.Pipe((shift,))
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxo.apply_op(pipe, None, q)
    Quantity(Array([1, 2, 3], dtype=int64), unit='km')

    """
    result = x
    for sub_op in op.operators:
        result = api.apply_op(sub_op, tau, result, **kw)
    return result


# ===================================================================
# Simplification


@plum.dispatch
def simplify(op: Pipe, /) -> AbstractOperator:
    """Simplify a Pipe operator.

    Examples
    --------
    >>> import coordinax as cx

    >>> shift = cx.ops.Translate.from_([1, 2, 3], "km")
    >>> identity = cx.ops.Identity()
    >>> pipe = cx.ops.Pipe((shift, identity))
    >>> pipe
    Pipe(( Translate(...), Identity() ))

    >>> cx.ops.simplify(pipe)
    Translate(...)

    """
    # TODO: figure out how to do pairwise simplifications
    # Remove identity operators
    simplified_ops = tuple(o for o in op.operators if not isinstance(o, Identity))
    # If no operators remain, return identity
    if not simplified_ops:
        return Identity()  # type: ignore[no-untyped-call]
    # If only one operator remains, return it
    if len(simplified_ops) == 1:
        return simplified_ops[0]
    # Otherwise, return simplified pipe
    return replace(op, operators=simplified_ops)
