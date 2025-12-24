"""Sequence of Operators."""

__all__ = ("Pipe",)

from dataclasses import replace

from typing import Any, final

import equinox as eqx
import plum
import wadler_lindig as wl

import unxt as u

from .base import AbstractOperator
from .composite import AbstractCompositeOperator
from .identity import Identity
from coordinax._src.api import apply_op, simplify
from coordinax._src.custom_types import CsDict


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

    >>> shift = cx.ops.Translate.from_([1, 2, 3], "km")
    >>> boost = cx.ops.Boost.from_([1, 2, 3], "km/s")
    >>> pipe = cx.ops.Pipe((shift, boost))
    >>> pipe
    Pipe((Translate(...), Boost(...)))

    A pipe can also be constructed by ``|``:

    >>> pipe2 = shift | boost
    >>> pipe2
    Pipe((Translate(...), Boost(...)))

    The pipe can be simplified. For this example, we add an identity operator to
    the sequence and simplify, which will remove the identity operator.

    >>> pipe3 = pipe2 | cx.ops.Identity()
    >>> pipe3
    Pipe((Translate(...), Boost(...), Identity()))

    >>> cx.ops.simplify(pipe3)
    Pipe((Translate(...), Boost(...)))

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
        Pipe((Translate(...), Boost(...), Translate(...), Boost(...)))

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

        >>> shift = cxo.Translate.from_([1, 2, 3], "km")
        >>> boost = cxo.Boost.from_([1, 2, 3], "km/s")
        >>> pipe = cxo.Pipe((shift, boost))
        >>> print(wl.pdoc(pipe))

        """
        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
        kwargs.setdefault("use_short_name", True)
        kwargs.setdefault("named_unit", False)

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


@plum.dispatch(precedence=1)
def operate(
    self: AbstractCompositeOperator, tau: Any, /, arg: object, **kw: object
) -> object:
    """Apply the operators in a sequence."""
    for op in self.operators:
        arg = op(tau, arg, **kw)
    return arg


@plum.dispatch(precedence=1)
def operate(
    self: AbstractCompositeOperator, tau: Any, /, *args: object, **kw: object
) -> tuple[object, ...]:
    """Apply the operators in a sequence."""
    for op in self.operators:
        args = op(tau, *args, **kw)
    return args


# ===================================================================
# apply_op for Pipe


@plum.dispatch
def apply_op(
    op: Pipe, tau: Any, x: CsDict, /, *, role: Any = None, at: Any = None
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
        result = apply_op(sub_op, tau, result, role=role, at=at)
    return result


@plum.dispatch
def apply_op(
    op: Pipe, tau: Any, x: u.AbstractQuantity, /, *, role: Any = None, at: Any = None
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
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    """
    result = x
    for sub_op in op.operators:
        result = apply_op(sub_op, tau, result, role=role, at=at)
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
    Pipe((Translate(...), Identity()))

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
