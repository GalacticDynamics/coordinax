"""Base classes for operators on coordinates and potentials."""

__all__: list[str] = []

import functools

from plum import dispatch

from .base import AbstractOperator
from .identity import Identity
from .pipe import Pipe


@dispatch(precedence=-1)  # very low priority
def simplify_op(op: AbstractOperator, /) -> AbstractOperator:
    """Return the operator unchanged.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> cx.ops.simplify_op(op) is op
    True

    """
    return op


@dispatch  # very low priority
def simplify_op(op: Identity, /) -> Identity:
    """Return the operator unchanged.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.Identity()
    >>> cx.ops.simplify_op(op) is op
    True

    """
    return op


@dispatch
def simplify_op(seq: Pipe, /) -> AbstractOperator:
    """Simplify a sequence of Operators.

    This simplifies the sequence of operators by removing any that reduce to
    the Identity operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> shift = cx.ops.GalileanSpatialTranslation(u.Quantity([1, 2, 3], "km"))
    >>> boost = cx.ops.GalileanBoost(u.Quantity([10, 20, 30], "km/s"))

    >>> seq = shift | cx.ops.Identity() | boost
    >>> seq
    Pipe((
        GalileanSpatialTranslation(...), Identity(), GalileanBoost(...)
    ))

    >>> cx.ops.simplify_op(seq)
    Pipe(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    """
    # TODO: more sophisticated operator fusion. This just applies pair-wise
    # simplification.
    return functools.reduce(simplify_op, seq.operators, Identity())


# ======================================================================
# 2-Operator Simplifications


@dispatch(precedence=-1)
def simplify_op(op1: AbstractOperator, op2: AbstractOperator, /) -> Pipe:
    """Simplify two operators into a sequence.

    Examples
    --------
    >>> import coordinax as cx

    >>> op1 = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> op2 = cx.ops.GalileanBoost.from_([0, 1, 0], "m/s")
    >>> cx.ops.simplify_op(op1, op2)
    Pipe((
        GalileanSpatialTranslation(CartesianPos3D( ... )),
        GalileanBoost(CartesianVel3D( ... ))
    ))

    """
    return Pipe((op1, op2))


@dispatch(precedence=1)
def simplify_op(op1: AbstractOperator, op2: Identity) -> AbstractOperator:
    """Simplify an operator with the identity.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> cx.ops.simplify_op(op, cx.ops.Identity())
    GalileanSpatialTranslation(...)

    """
    return op1


@dispatch
def simplify_op(op1: Identity, op2: AbstractOperator) -> AbstractOperator:
    """Simplify an operator with the identity.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> cx.ops.simplify_op(cx.ops.Identity(), op)
    GalileanSpatialTranslation(...)

    """
    return op2


@dispatch
def simplify_op(op1: Pipe, op2: AbstractOperator) -> Pipe:
    """Simplify two sequences of operators by concatenating them.

    Examples
    --------
    >>> import coordinax as cx

    >>> sop1 = cx.ops.Pipe((cx.ops.Identity(), cx.ops.Identity()))
    >>> sop2 = cx.ops.Pipe((cx.ops.Identity(),))
    >>> cx.ops.simplify_op(sop1, sop2)
    Pipe((Identity(), Identity(), Identity()))

    """
    return op1 | op2


@dispatch
def simplify_op(op1: Pipe, op2: Pipe) -> Pipe:
    """Simplify two sequences of operators by concatenating them.

    Examples
    --------
    >>> import coordinax as cx

    >>> sop1 = cx.ops.Pipe((cx.ops.Identity(), cx.ops.Identity()))
    >>> sop2 = cx.ops.Pipe((cx.ops.Identity(),))
    >>> cx.ops.simplify_op(sop1, sop2)
    Pipe((Identity(), Identity(), Identity()))

    """
    return Pipe(op1.operators + op2.operators)
