"""Role-specialized additive operators.

This module defines primitive operators that are specialized by role:

- ``Translate``: Translation of points (acts on ``Point`` role)
- ``Boost``: Velocity boost (acts on ``Vel`` role)
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

__all__ = ("Translate",)


from dataclasses import KW_ONLY

from collections.abc import Callable
from typing import Any, Protocol, Union, final, runtime_checkable

import equinox as eqx
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax._src.roles as cxr
from .base import AbstractOperator, Neg, eval_op
from .identity import Identity
from .pipe import Pipe
from .utils import _require_role_for_pdict
from coordinax._src.custom_types import CsDict


@runtime_checkable
class CanAddandNeg(Protocol):
    """Protocol for classes that support addition and negation."""

    def __add__(self, other: Any, /) -> Any: ...
    def __neg__(self, /) -> Any: ...


#####################################################################
# Translate: Translation of points


@final
class Translate(AbstractOperator):
    r"""Operator for translating points.

    This operator applies a spatial displacement to point-role vectors.
    Mathematically, it implements:

    $$
    p' = p + \Delta p
    $$

    where $p$ is a point and $\Delta p$ is the displacement vector.

    Parameters
    ----------
    delta : Vector | Quantity | CsDict | Callable
        The displacement to apply. Must have length dimension.
        If callable, will be evaluated at the time parameter ``tau``.

    Notes
    -----
    - Only applicable to ``Point`` role vectors
    - Raises ``TypeError`` when applied to other roles (e.g., ``Vel``)
    - The ``delta`` is interpreted as a ``Pos``-role displacement internally

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    Create a translation operator:

    >>> shift = cxo.Translate.from_([1, 2, 3], "km")
    >>> shift
    Translate(Q(...[3], 'km'))

    The inverse negates the displacement:

    >>> shift.inverse
    Translate(Q(...[3], 'km'))

    Time-dependent translation:

    >>> moving = cxo.Translate(lambda t: u.Q(t.ustrip("s"), "km"))

    See Also
    --------
    Boost : Velocity offset operator
    AccelShift : Acceleration offset operator

    """

    delta: CanAddandNeg | Callable[[Any], Any]
    """The displacement (length units)."""

    _: KW_ONLY

    right_add: bool = eqx.field(default=True, static=True)

    # -------------------------------------------

    @property
    def inverse(self) -> "Translate":
        """The inverse translation (negated displacement).

        Examples
        --------
        >>> import coordinax.ops as cxo

        >>> shift = cxo.Translate.from_([1, 2, 3], "km")
        >>> shift.inverse
        Translate(Q(...[3], 'km'))

        """
        delta = self.delta
        inv = -delta if (not callable(delta) or isinstance(delta, Neg)) else Neg(delta)
        return Translate(inv)

    # -------------------------------------------
    # Python API

    def __add__(self, other: object, /) -> Union["Translate", Pipe]:
        """Combine two translations into a single translation."""
        if not isinstance(other, Translate):
            return NotImplemented

        if not callable(self.delta) and not callable(other.delta):
            return Translate(self.delta + other.delta)
        return Pipe((self, other))

    def __neg__(self, /) -> "Translate":
        """Return negative of the translation."""
        return self.inverse


@Translate.from_.dispatch
def from_(cls: type[Translate], q: u.AbstractQuantity, /) -> Translate:
    """Construct a Translate from a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> cxo.Translate.from_(u.Q([1, 2, 3], "km"))
    Translate(...)

    """
    return Translate(q)


@plum.dispatch
def simplify(op: Translate, /, **kwargs: Any) -> Translate | Identity:
    """Simplify a Translate operator.

    A translation with zero delta simplifies to Identity.

    Examples
    --------
    >>> import coordinax.ops as cxo

    >>> op = cxo.Translate.from_([1, 2, 3], "km")
    >>> cxo.simplify(op)
    Translate(...)

    >>> op = cxo.Translate.from_([0, 0, 0], "km")
    >>> cxo.simplify(op)
    Identity()

    """
    if jnp.allclose(u.ustrip(AllowValue, op.delta), 0, **kwargs):
        return Identity()
    return op


# =====================================


def _apply_add_to_cdict(op_delta: Any, x: CsDict) -> CsDict:
    """Apply an add-type operator's delta to a CsDict.

    Handles three cases:
    1. delta is a dict: add component-wise by key
    2. delta has .data attribute (Vector-like): use its data dict
    3. delta is a 1D array-like Quantity: index into components
    4. delta is a scalar: broadcast add to all components

    """
    delta = op_delta

    # If delta is a dict, add component-wise
    if isinstance(delta, dict):
        return {k: x[k] + delta.get(k, 0) for k in x}

    # If delta has a .data attribute (Vector-like), use its data
    if hasattr(delta, "data"):
        delta_data = delta.data
        return {k: x[k] + delta_data.get(k, 0) for k in x}

    # If delta is a 1D Quantity array, index into components
    # This handles cases like Boost.delta = Q([vx, vy, vz], 'km/s')
    if isinstance(delta, u.AbstractQuantity) and delta.ndim >= 1:
        keys = list(x.keys())
        if len(keys) == delta.shape[0]:
            return {k: x[k] + delta[i] for i, k in enumerate(keys)}

    # Otherwise, broadcast add (scalar delta to all components)
    return {k: v + delta for k, v in x.items()}


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    x: u.AbstractQuantity,
    /,
    *,
    role: Any = None,
    at: Any = None,
) -> u.AbstractQuantity:
    """Apply Translate to a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> shift = cxo.Translate.from_([1, 2, 3], "km")
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxo.apply_op(shift, None, q)
    Quantity(...[1, 2, 3]...'km')

    """
    op_eval: Translate = eval_op(op, tau)
    delta = op_eval.delta
    if op.right_add:
        return x + delta
    return delta + x


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    x: CsDict,
    /,
    *,
    role: cxr.AbstractRole | None = None,
    at: Any = None,
) -> CsDict:
    """Apply Translate to a CsDict.

    Parameters
    ----------
    op : Translate
        The translation operator.
    tau : Any
        Time parameter for time-dependent operators.
    x : CsDict
        The point data to translate.
    role : AbstractRole
        Must be ``Point``. Required for CsDict inputs.
    at : Any
        Unused for Translate.

    Returns
    -------
    CsDict
        The translated point data.

    Raises
    ------
    TypeError
        If ``role`` is not ``Point`` or not provided.

    """
    _require_role_for_pdict(role)
    del at  # unused

    # Validate that role is Point
    if not isinstance(role, cxr.Point):
        msg = (
            f"Translate can only act on Point role, got {role}. "
            "Use Boost for Vel or AccelShift for Acc."
        )
        raise TypeError(msg)

    # Materialize and apply
    op_eval: Translate = eval_op(op, tau)
    return _apply_add_to_cdict(op_eval.delta, x)
