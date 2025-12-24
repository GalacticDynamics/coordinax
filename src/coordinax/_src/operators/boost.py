"""Role-specialized additive operators.

This module defines primitive operators that are specialized by role:

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

__all__ = ("Boost", "AccelShift")


from dataclasses import KW_ONLY

from collections.abc import Callable
from typing import Any, Union, final

import equinox as eqx
import plum

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless
from unxt.quantity import AllowValue

import coordinax._src.roles as cxr
from .base import AbstractOperator, Neg, eval_op
from .identity import Identity
from .pipe import Pipe
from .translate import CanAddandNeg
from .utils import _require_role_for_pdict
from coordinax._src.custom_types import CsDict


@final
class Boost(AbstractOperator):
    r"""Operator for velocity boosts (Galilean frame transformations).

    This operator implements a constant-velocity boost with well-defined
    actions on all kinematic roles:

    **Point (position) action:**

    $$ x'(\tau) = x(\tau) + v_0 \cdot (\tau - \tau_0) $$

    The boost translates points by an amount proportional to elapsed time.

    **Displacement (Pos) action:**

    $$ \Delta x' = \Delta x $$

    Displacements at the same $\tau$ are invariant under Galilean boosts.

    **Velocity action:**

    $$ v' = v + v_0 $$

    **Acceleration action:**

    $$ a' = a $$

    Accelerations are invariant under constant boosts.

    Parameters
    ----------
    delta : Vector | Quantity | CsDict | Callable
        The velocity offset $v_0$ to apply. Must have speed dimension.
        If callable, will be evaluated at the time parameter ``tau``.
    tau0 : Quantity['time'] | None
        Reference epoch for the boost. Defaults to 0.
        The boost translation for Point role is $v_0 \cdot (\tau - \tau_0)$.

    Notes
    -----
    The ``tau0`` parameter defaults to zero, meaning the point-action
    translation is simply $v_0 \cdot \tau$. To change reference frames at a
    specific epoch, pass an explicit ``tau0``.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    Create a boost operator:

    >>> boost = cxo.Boost.from_([100, 0, 0], "km/s")
    >>> boost
    Boost(Q(...[3], 'km / s'), ...)

    With explicit reference epoch:

    >>> boost_epoch = cxo.Boost.from_([100, 0, 0], "km/s", tau0=u.Q(10, "yr"))
    >>> boost_epoch.tau0
    Quantity...10...'yr')

    In a pipeline (shift + boost):

    >>> shift = cxo.Translate.from_([1, 0, 0], "km")
    >>> pipeline = shift | boost

    See Also
    --------
    Translate : Point translation operator
    AccelShift : Acceleration offset operator

    """

    delta: CanAddandNeg | Callable[[Any], Any]
    """The velocity offset $v_0$ (speed units)."""

    tau0: u.AbstractQuantity = eqx.field(
        default_factory=lambda: u.StaticQuantity(0, "s"),
        converter=Unless(u.AbstractQuantity, u.Q.from_),
    )
    r"""Reference epoch $\tau_0$ for the boost (time units)."""

    _: KW_ONLY

    right_add: bool = eqx.field(default=True, static=True)

    # -------------------------------------------

    @property
    def inverse(self) -> "Boost":
        """The inverse boost (negated velocity offset).

        Examples
        --------
        >>> import coordinax.ops as cxo

        >>> boost = cxo.Boost.from_([100, 0, 0], "km/s")
        >>> boost.inverse
        Boost(Q(...[3], 'km / s'), ...)

        """
        delta = self.delta
        inv = -delta if (not callable(delta) or isinstance(delta, Neg)) else Neg(delta)
        return Boost(inv, tau0=self.tau0)

    # -------------------------------------------
    # Python API

    def __add__(self, other: object, /) -> Union["Boost", Pipe]:
        """Combine two boosts into a single boost."""
        if not isinstance(other, Boost):
            return NotImplemented

        if not callable(self.delta) and not callable(other.delta):
            # Use self.tau0 for the combined boost
            return Boost(self.delta + other.delta, tau0=self.tau0)
        return Pipe((self, other))

    def __neg__(self, /) -> "Boost":
        """Return negative of the boost."""
        return self.inverse


@Boost.from_.dispatch(precedence=1)
def from_(
    cls: type[Boost],
    x: list,  # type: ignore[type-arg]
    unit: str,
    /,
    *,
    tau0: u.AbstractQuantity = u.StaticQuantity(0, "s"),
) -> Boost:
    """Construct a Boost from a list and unit string.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> cxo.Boost.from_([100, 0, 0], "km/s")
    Boost(Q(...[3], 'km / s'), ...)

    With explicit reference epoch:

    >>> cxo.Boost.from_([100, 0, 0], "km/s", tau0=u.Q(1, "yr"))
    Boost(Q(...[3], 'km / s'), ...)

    """
    return Boost(u.Q(x, unit), tau0=tau0)


@Boost.from_.dispatch
def from_(
    cls: type[Boost],
    delta: Any,
    /,
    *,
    tau0: u.AbstractQuantity = u.StaticQuantity(0, "s"),
) -> Boost:
    """Construct a Boost from a Vector or similar."""
    return Boost(delta, tau0=tau0)


@Boost.from_.dispatch
def from_(
    cls: type[Boost],
    q: u.AbstractQuantity,
    /,
    *,
    tau0: u.AbstractQuantity = u.StaticQuantity(0, "s"),
) -> Boost:
    """Construct a Boost from a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> cxo.Boost.from_(u.Q([100, 0, 0], "km/s"))
    Boost(...)

    With explicit reference epoch:

    >>> cxo.Boost.from_(u.Q([100, 0, 0], "km/s"), tau0=u.Q(1, "yr"))
    Boost(Q(...[3], 'km / s'), ...)

    """
    return Boost(q, tau0=tau0)


@plum.dispatch
def simplify(op: Boost, /, **kwargs: Any) -> Boost | Identity:
    """Simplify a Boost operator.

    A boost with zero delta simplifies to Identity.

    Examples
    --------
    >>> import coordinax.ops as cxo

    >>> op = cxo.Boost.from_([100, 0, 0], "km/s")
    >>> cxo.simplify(op)
    Boost(...)

    >>> op = cxo.Boost.from_([0, 0, 0], "km/s")
    >>> cxo.simplify(op)
    Identity()

    """
    if jnp.allclose(u.ustrip(AllowValue, op.delta), 0, **kwargs):
        return Identity()
    return op


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    x: u.AbstractQuantity,
    /,
    *,
    role: Any = None,
    at: Any = None,
) -> u.AbstractQuantity:
    """Apply Boost to a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> boost = cxo.Boost.from_([100, 0, 0], "km/s")
    >>> v = u.Q([0, 0, 0], "km/s")
    >>> cxo.apply_op(boost, None, v)
    Quantity(...[100..., 0..., 0...]...'km / s')

    """
    op_eval: Boost = eval_op(op, tau)
    delta = op_eval.delta
    if op.right_add:
        return x + delta
    return delta + x


def _compute_boost_displacement(
    op: Boost, tau: Any, tau0: u.AbstractQuantity
) -> u.AbstractQuantity:
    r"""Compute the displacement $v_0 \cdot (\tau - \tau_0)$ for boost on Point.

    Parameters
    ----------
    op : Boost
        The boost operator (with velocity delta).
    tau : Any
        Current time parameter.
    tau0 : Quantity['time']
        Reference epoch.

    Returns
    -------
    Quantity['length']
        The displacement to apply.

    """
    v0 = eval_op(op, tau).delta
    dt = tau - tau0
    return v0 * dt


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    x: CsDict,
    /,
    *,
    role: cxr.AbstractRole | None = None,
    at: Any = None,
) -> CsDict:
    r"""Apply Boost to a CsDict.

    The action depends on the role:

    - **Point**: $x'(\tau) = x(\tau) + v_0 \cdot (\tau - \tau_0)$
    - **Pos**: Identity (displacement invariant under boost)
    - **Vel**: $v' = v + v_0$
    - **Acc**: Identity (constant boost leaves acceleration invariant)

    Parameters
    ----------
    op : Boost
        The boost operator.
    tau : Any
        Time parameter. Required for Point role to compute $\tau - \tau_0$.
    x : CsDict
        The data to boost.
    role : AbstractRole
        The geometric role. Required for CsDict inputs.
    at : Any
        Base point for non-Euclidean charts.

    Returns
    -------
    CsDict
        The transformed data.

    Raises
    ------
    TypeError
        If ``role`` is not provided.

    """
    _require_role_for_pdict(role)
    del at  # may be needed for non-Euclidean; currently unused

    # Point role: translate by v0 * (tau - tau0)
    if isinstance(role, cxr.Point):
        displacement = _compute_boost_displacement(op, tau, op.tau0)
        return _apply_add_to_pdict(displacement, x)

    # Pos role: identity (displacement invariant)
    if isinstance(role, cxr.Pos):
        return x

    # Vel role: add boost velocity
    if isinstance(role, cxr.Vel):
        op_eval: Boost = eval_op(op, tau)
        return _apply_add_to_pdict(op_eval.delta, x)

    # Acc role: identity (constant boost)
    if isinstance(role, cxr.Acc):
        return x

    # Unknown role
    msg = f"Boost does not know how to act on role {role}."
    raise TypeError(msg)


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
