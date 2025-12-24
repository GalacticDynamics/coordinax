"""Role-specialized additive operators.

This module defines primitive operators that are specialized by role:

- ``AccelShift``: Acceleration offset (acts on ``Acc`` role)

These operators provide:

- Type-safe operations constrained to specific geometric roles
- Clear error messages when applied to incompatible roles
- Composability via ``Pipe`` for reference-frame transformations

See Also
--------
coordinax.ops.apply_op : Apply an operator to an input
coordinax.ops.eval_op : Materialize time-dependent parameters
coordinax.ops.Pipe : Compose operators into a pipeline

"""

__all__ = ("AccelShift",)


from dataclasses import KW_ONLY

from collections.abc import Callable
from typing import Any, Union, final

import equinox as eqx
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax._src.roles as cxr
from .base import AbstractOperator, Neg, eval_op
from .identity import Identity
from .pipe import Pipe
from .translate import CanAddandNeg
from .utils import _require_role_for_pdict
from coordinax._src.custom_types import CsDict

#####################################################################
# AccelShift: Acceleration offset


@final
class AccelShift(AbstractOperator):
    r"""Operator for acceleration offsets.

    This operator applies an acceleration offset to acceleration-role vectors.
    Mathematically, it implements:

    $$
    a' = a + \Delta a
    $$

    where $a$ is an acceleration and $\Delta a$ is the offset.

    Parameters
    ----------
    delta : Vector | Quantity | CsDict | Callable
        The acceleration offset to apply. Must have acceleration dimension.
        If callable, will be evaluated at the time parameter ``tau``.

    Notes
    -----
    - Only applicable to ``Acc`` role vectors
    - Raises ``TypeError`` when applied to other roles
    - When applied to a ``FiberPoint``, acts on the acceleration field

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    Create an acceleration shift operator:

    >>> accel = cxo.AccelShift.from_([0, 0, -9.8], "m/s^2")
    >>> accel
    AccelShift(Q(f64[3], 'm / s2'))

    See Also
    --------
    Translate : Point translation operator
    Boost : Velocity boost operator

    """

    delta: CanAddandNeg | Callable[[Any], Any]
    """The acceleration offset (acceleration units)."""

    _: KW_ONLY

    right_add: bool = eqx.field(default=True, static=True)

    # -------------------------------------------

    @property
    def inverse(self) -> "AccelShift":
        """The inverse acceleration shift (negated offset).

        Examples
        --------
        >>> import coordinax.ops as cxo

        >>> accel = cxo.AccelShift.from_([0, 0, -9.8], "m/s^2")
        >>> accel.inverse
        AccelShift(Q(...[3], 'm / s2'))

        """
        delta = self.delta
        inv = -delta if (not callable(delta) or isinstance(delta, Neg)) else Neg(delta)
        return AccelShift(inv)

    # -------------------------------------------
    # Python API

    def __add__(self, other: object, /) -> Union["AccelShift", Pipe]:
        """Combine two acceleration shifts into a single shift."""
        if not isinstance(other, AccelShift):
            return NotImplemented

        if not callable(self.delta) and not callable(other.delta):
            return AccelShift(self.delta + other.delta)
        return Pipe((self, other))

    def __neg__(self, /) -> "AccelShift":
        """Return negative of the acceleration shift."""
        return self.inverse


@AccelShift.from_.dispatch
def from_(cls: type[AccelShift], delta: Any, /) -> AccelShift:
    """Construct an AccelShift from a Vector or similar."""
    return AccelShift(delta)


@AccelShift.from_.dispatch
def from_(cls: type[AccelShift], q: u.AbstractQuantity, /) -> AccelShift:
    """Construct an AccelShift from a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> cxo.AccelShift.from_(u.Q([0, 0, -9.8], "m/s^2"))
    AccelShift(...)

    """
    return AccelShift(q)


@plum.dispatch
def simplify(op: AccelShift, /, **kwargs: Any) -> AccelShift | Identity:
    """Simplify an AccelShift operator.

    An acceleration shift with zero delta simplifies to Identity.

    Examples
    --------
    >>> import coordinax.ops as cxo

    >>> op = cxo.AccelShift.from_([0, 0, -9.8], "m/s^2")
    >>> cxo.simplify(op)
    AccelShift(...)

    >>> op = cxo.AccelShift.from_([0, 0, 0], "m/s^2")
    >>> cxo.simplify(op)
    Identity()

    """
    if jnp.allclose(u.ustrip(AllowValue, op.delta), 0, **kwargs):
        return Identity()
    return op


@plum.dispatch
def apply_op(
    op: AccelShift,
    tau: Any,
    x: CsDict,
    /,
    *,
    role: cxr.AbstractRole | None = None,
    at: Any = None,
) -> CsDict:
    """Apply AccelShift to a CsDict.

    Parameters
    ----------
    op : AccelShift
        The acceleration shift operator.
    tau : Any
        Time parameter for time-dependent operators.
    x : CsDict
        The acceleration data to shift.
    role : AbstractRole
        Must be ``Acc``. Required for CsDict inputs.
    at : Any
        Base point for non-Euclidean charts.

    Returns
    -------
    CsDict
        The shifted acceleration data.

    Raises
    ------
    TypeError
        If ``role`` is not ``Acc`` or not provided.

    """
    _require_role_for_pdict(role)
    del at  # may be needed for non-Euclidean; currently unused

    # Validate that role is Acc
    if not isinstance(role, cxr.Acc):
        msg = (
            f"AccelShift can only act on Acc role, got {role}. "
            "Use Translate for Point or Boost for Vel."
        )
        raise TypeError(msg)

    # Materialize and apply
    op_eval: AccelShift = eval_op(op, tau)
    return _apply_add_to_pdict(op_eval.delta, x)


@plum.dispatch
def apply_op(
    op: AccelShift,
    tau: Any,
    x: u.AbstractQuantity,
    /,
    *,
    role: Any = None,
    at: Any = None,
) -> u.AbstractQuantity:
    """Apply AccelShift to a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> accel = cxo.AccelShift.from_([0, 0, -9.8], "m/s^2")
    >>> a = u.Q([0, 0, 0], "m/s^2")
    >>> cxo.apply_op(accel, None, a)
    Quantity(...[ 0. ,  0. , -9.8]...'m / s2')

    """
    op_eval: AccelShift = eval_op(op, tau)
    delta = op_eval.delta
    if op.right_add:
        return x + delta
    return delta + x
