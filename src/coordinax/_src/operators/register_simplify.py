"""Base classes for operators on coordinates and potentials."""

__all__: list[str] = []

import functools

from plum import dispatch

from .base import AbstractOperator
from .identity import Identity
from .sequence import Sequence


@dispatch(precedence=-1)  # very low priority
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


@dispatch  # very low priority
def simplify_op(op: Identity, /) -> Identity:
    """Return the operator unchanged.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> op = cxo.Identity()
    >>> cxo.simplify_op(op) is op
    True

    """
    return op


@dispatch
def simplify_op(seq: Sequence, /) -> AbstractOperator:
    """Simplify a sequence of Operators.

    This simplifies the sequence of operators by removing any that reduce to
    the Identity operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.operators as co

    >>> shift = co.GalileanSpatialTranslation(u.Quantity([1, 2, 3], "kpc"))
    >>> boost = co.GalileanBoost(u.Quantity([10, 20, 30], "km/s"))

    >>> seq = shift | co.Identity() | boost
    >>> seq
    Sequence((
        GalileanSpatialTranslation(...), Identity(), GalileanBoost(...)
    ))

    >>> co.simplify_op(seq)
    Sequence(( GalileanSpatialTranslation(...), GalileanBoost(...) ))

    """
    # TODO: more sophisticated operator fusion. This just applies pair-wise
    # simplification.
    return functools.reduce(simplify_op, seq.operators, Identity())


# ======================================================================
# 2-Operator Simplifications


@dispatch(precedence=-1)
def simplify_op(op1: AbstractOperator, op2: AbstractOperator, /) -> Sequence:
    """Simplify two operators into a sequence.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> op1 = cxo.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> op2 = cxo.GalileanBoost.from_([0, 1, 0], "m/s")
    >>> cxo.simplify_op(op1, op2)
    Sequence((
        GalileanSpatialTranslation(CartesianPos3D( ... )),
        GalileanBoost(CartesianVel3D( ... ))
    ))

    """
    return Sequence((op1, op2))


@dispatch(precedence=1)
def simplify_op(op1: AbstractOperator, op2: Identity) -> AbstractOperator:
    """Simplify an operator with the identity.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> op = cxo.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> cxo.simplify_op(op, cxo.Identity())
    GalileanSpatialTranslation(...)

    """
    return op1


@dispatch
def simplify_op(op1: Identity, op2: AbstractOperator) -> AbstractOperator:
    """Simplify an operator with the identity.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> op = cxo.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> cxo.simplify_op(cxo.Identity(), op)
    GalileanSpatialTranslation(...)

    """
    return op2


@dispatch
def simplify_op(op1: Sequence, op2: AbstractOperator) -> Sequence:
    """Simplify two sequences of operators by concatenating them.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> sop1 = cxo.Sequence((cxo.Identity(), cxo.Identity()))
    >>> sop2 = cxo.Sequence((cxo.Identity(),))
    >>> cxo.simplify_op(sop1, sop2)
    Sequence((Identity(), Identity(), Identity()))

    """
    return op1 | op2


@dispatch
def simplify_op(op1: Sequence, op2: Sequence) -> Sequence:
    """Simplify two sequences of operators by concatenating them.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> sop1 = cxo.Sequence((cxo.Identity(), cxo.Identity()))
    >>> sop2 = cxo.Sequence((cxo.Identity(),))
    >>> cxo.simplify_op(sop1, sop2)
    Sequence((Identity(), Identity(), Identity()))

    """
    return Sequence(op1.operators + op2.operators)
