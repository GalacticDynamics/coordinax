"""Role-specialized additive operators.

This module defines primitive operators that are specialized by role:

- ``Boost``: Velocity boost
  (acts on ``Point``, ``PhysDisp``, ``PhysVel``, ``PhysAcc`` roles)

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

__all__ = ("Boost",)


from collections.abc import Callable
from jaxtyping import Array, ArrayLike
from typing import Any, Final, final

import equinox as eqx
import jax
import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax.api as cxapi
import coordinax.charts as cxc
import coordinax.roles as cxr
from .add import AbstractAdd
from .base import eval_op
from .identity import Identity
from coordinax._src.custom_types import OptUSys, Unit
from coordinax._src.utils import pack_uniform_unit
from coordinax.api import CsDict

SEC = u.unit("s")


@final
class Boost(AbstractAdd):
    r"""Operator for Galilean velocity boosts.

    A boost represents a reference-frame velocity offset operator. It is a
    time-parameterized affine transformation acting on events/points as well
    as on physical tangent quantities.

    Per the spec, it has distinct actions on different roles:

    **Point (event/position)**: Time-dependent translation

        $$ p'(\tau) = p(\tau) + (\tau - \tau_0) \Delta v $$

        where $\tau_0$ is the reference epoch (assumed to be 0).

    **Pos (physical displacement)**: Identity (invariant under Galilean boost)

        $$ \Delta x' = \Delta x $$

    **Vel (velocity)**: Adds the boost velocity

        $$ v'(\tau) = v(\tau) + \Delta v $$

    **Acc (acceleration)**:
      - If $\Delta v$ is time-independent: Identity
      - If $\Delta v(\tau)$ is time-dependent: Adds time derivative

        $$ a'(\tau) = a(\tau) + \frac{d}{d\tau}\Delta v(\tau) $$

    Parameters
    ----------
    delta : CsDict | Callable[[tau], CsDict]
        The velocity offset to apply. Must have velocity dimension.
        If callable, will be evaluated at the time parameter ``tau``.
    chart : AbstractChart
        Chart in which delta is expressed.

    Notes
    -----
    - Applicable to ``Point``, ``PhysDisp``, ``PhysVel``, and ``PhysAcc`` roles
    - For Point role, requires tau to be a time Quantity when delta is nonzero
    - The ``delta`` is interpreted as a velocity offset

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxop

    Create a boost operator:

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> boost
    Boost(
        {'x': Q(100, 'km / s'), 'y': Q(0, 'km / s'), 'z': Q(0, 'km / s')},
        chart=Cart3D()
    )

    Apply to velocity:

    >>> v = u.Q([10, 20, 30], "km/s")
    >>> cxop.apply_op(boost, None, cxr.phys_vel, cxc.cart3d, v)
    Quantity(Array([110,  20,  30], dtype=int64), unit='km / s')

    Apply to point (requires tau):

    >>> p = u.Q([0, 0, 0], "km")
    >>> tau = u.Q(1.0, "s")
    >>> cxop.apply_op(boost, tau, cxr.point, cxc.cart3d, p)
    Quantity(Array([100.,   0.,   0.], dtype=float64, ...), unit='km')

    See Also
    --------
    Translate : Position translation operator

    """

    # delta, chart, and right_add inherited from AbstractAdd

    # inverse, __add__, and __neg__ inherited from AbstractAdd


@Boost.from_.dispatch
def from_(cls: type[Boost], obj: Boost, /) -> Boost:
    """Construct a Boost from another Boost.

    Examples
    --------
    >>> import coordinax as cx
    >>> boost1 = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> cxop.Boost.from_(boost1) is boost1
    True

    """
    return obj


@Boost.from_.dispatch
def from_(cls: type[Boost], q: u.AbstractQuantity, /) -> Boost:
    """Construct a Boost from a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxop

    >>> cxop.Boost.from_(u.Q([100, 0, 0], "km/s"))
    Boost(
        {'x': Q(100, 'km / s'), 'y': Q(0, 'km / s'), 'z': Q(0, 'km / s')},
        chart=Cart3D()
    )

    """
    chart = cxapi.guess_chart(q)
    x = cxapi.cdict(q, chart)
    return cls(x, chart=chart)


@plum.dispatch
def simplify(op: Boost, /, **kw: Any) -> Boost | Identity:
    """Simplify a Boost operator.

    A boost with zero dv simplifies to Identity.

    Examples
    --------
    >>> import coordinax.ops as cxop

    >>> op = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> cxop.simplify(op)
    Boost(...)

    >>> op = cxop.Boost.from_([0, 0, 0], "km/s")
    >>> cxop.simplify(op)
    Identity()

    """
    # Can't simplify callable
    if callable(op.delta):
        return op
    # Compute if the boost is an identity operation
    is_zero = jtu.all(
        jtu.map(lambda v: jnp.allclose(u.ustrip(AllowValue, v), 0, **kw), op.delta)
    )
    if is_zero:
        return Identity()  # type: ignore[no-untyped-call]
    return op


# ============================================================================
# Helper functions


def _require_tau_time(tau: Any, /) -> tuple[Array, Unit]:
    """Return (tau_value, tau_unit), requiring tau to be a time quantity."""
    # We require a Quantity-like tau with time dimension so we can form d/dtau.
    tau = eqx.error_if(
        tau,
        not u.quantity.is_any_quantity(tau),
        "Boost requires tau as a time Quantity when delta is time-dependent",
    )
    unit = u.unit_of(tau)
    tau = eqx.error_if(
        tau,
        u.dimension_of(unit) != u.dimension_of(SEC),
        "Boost requires tau to have time dimension",
    )
    # Differentiate with respect to the numeric tau in its own unit.
    # Ensure float dtype for jacfwd
    return jnp.asarray(u.ustrip(unit, tau), dtype=float), unit


def _delta_values_fn(
    delta_fn: Callable[[Any], Any],
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    tau_unit: Unit,
    /,
) -> tuple[Callable[[Array], Array], Unit | None]:
    """Build a pure-JAX function returning packed delta values.

    Returns
    -------
    (f, unit)
        f: tau_value -> Array[... , n] of dv values in `chart.components` order.
        unit: common velocity unit returned by `pack_uniform_unit`.

    Notes
    -----
    We wrap `tau_value` back into a Quantity with unit `tau_unit` before calling
    the user-provided `delta_fn`.

    """
    keys = chart.components

    def f(tau_value: Array) -> Array:
        tau_q = u.Q(tau_value, tau_unit)
        d = delta_fn(tau_q)
        vals, _unit = pack_uniform_unit(d, keys=keys)
        # Ensure an array output for JAX differentiation.
        return jnp.asarray(vals)

    # One eager call to learn the unit.
    vals0, unit0 = pack_uniform_unit(delta_fn(u.Q(jnp.zeros(()), tau_unit)), keys=keys)
    del vals0

    return f, unit0


def _time_derivative_delta(
    op: Boost, tau: Any, /, order: int
) -> tuple[CsDict, cxc.AbstractChart[Any, Any]]:
    """Compute d^order/dtau^order delta(tau) as a CsDict in op.chart.

    Parameters
    ----------
    op : Boost
        The boost operator with callable delta.
    tau : Any
        Time parameter (must be a Quantity with time dimension).
    order
        1 for acceleration shift.

    Returns
    -------
    (deltas, chart)
        `deltas` has the same component keys as `op.chart.components`.
        Units are `velocity / time^order`.

    """
    if not callable(op.delta):
        raise TypeError("_time_derivative_delta requires callable delta")

    tau_value, tau_unit = _require_tau_time(tau)

    f, vel_unit = _delta_values_fn(op.delta, op.chart, tau_unit)

    if order == 1:
        df = jax.jacfwd(f)(tau_value)
        out_vals = df
        out_unit = None if vel_unit is None else (vel_unit / tau_unit)
    else:
        msg = f"unsupported derivative order={order}"
        raise ValueError(msg)

    # Unpack back into a component dict.
    comps = op.chart.components
    if out_unit is None:
        out = {k: out_vals[..., i] for i, k in enumerate(comps)}
    else:
        out = {k: u.Q(out_vals[..., i], out_unit) for i, k in enumerate(comps)}

    return out, op.chart


# ============================================================================
# apply_op dispatches

# ------------------------------
# ArrayLike dispatches


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    r"""Apply Boost to a Point-valued ArrayLike.

    Per spec: Boost acts on Point as a time-dependent translation:
    $$ p'(\tau) = p(\tau) + (\tau - \tau_0) * \Delta_v $$

    For constant boost, this is a translation by $(\tau - \tau_0) * \Delta_v$.
    Requires tau to be a time Quantity when delta is nonzero.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> tau = u.Q(1.0, "s")
    >>> usys = u.unitsystems.si
    >>> cxop.apply_op(boost, tau, cxr.point, cxc.cart3d, x, usys=usys)
    Array([100000., 0., 0.], dtype=float64)

    """
    del role, kw
    usys = eqx.error_if(usys, usys is None, "usys required")

    # For ArrayLike, we only support Cartesian charts.
    cart = chart.cartesian
    chart = eqx.error_if(
        chart, chart != cart, "Boost(Point, ArrayLike) requires a Cartesian chart"
    )

    x_arr = jnp.asarray(x)

    # Materialize dv
    op_eval = eval_op(op, tau)

    # Restrict to matching Cartesian charts for ArrayLike.
    op_chart = op.chart
    op_cart = op_chart.cartesian
    _ = eqx.error_if(
        op_chart,
        op_chart != op_cart or op_cart != chart,
        "Boost(Point, ArrayLike) requires op.chart to match the input "
        "Cartesian chart; use CsDict/Quantity for nontrivial chart conversions.",
    )

    dv_vals, dv_unit = pack_uniform_unit(op_eval.delta, keys=chart.components)

    # Check if dv is nonzero - if so, require tau to be a time Quantity
    if not jnp.allclose(dv_vals, 0):
        tau_value, _ = _require_tau_time(tau)

        # Convert dv to canonical velocity unit
        vel_unit = usys[u.dimension_of(u.unit("m/s"))]
        if dv_unit is not None:
            dv_vals = u.uconvert_value(vel_unit, dv_unit, dv_vals)

        # Compute displacement: delta_x = (tau - tau0) * delta_v
        # Assuming tau0 = 0 for simplicity (as per common convention)
        displacement = tau_value * dv_vals

        # Convert to position unit
        usys[u.dimension_of(u.unit("m"))]
        # displacement has units of time * velocity = length
        # dv_vals is already in vel_unit, so tau_value * dv_vals has correct dimension

        return x_arr + displacement if op_eval.right_add else displacement + x_arr
    # Zero boost: identity on points
    return x_arr


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysDisp,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Boost to a Pos-valued ArrayLike.

    Per spec: Galilean boost is identity on Pos (physical displacements).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> d = jnp.array([10.0, 20.0, 30.0])
    >>> cxop.apply_op(boost, None, cxr.phys_disp, cxc.cart3d, d)
    Array([10., 20., 30.], dtype=float64)

    """
    del op, tau, role, chart, usys, kw
    return jnp.asarray(x)


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysVel,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Boost to a Vel-valued ArrayLike.

    ArrayLike is interpreted as Cartesian velocity components in the canonical
    velocity unit from `usys`. This dispatch only supports the case `op.chart`
    matches the input Cartesian chart.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> v = jnp.array([10.0, 20.0, 30.0])
    >>> usys = u.unitsystems.si
    >>> cxop.apply_op(boost, None, cxr.phys_vel, cxc.cart3d, v, usys=usys)
    Array([1.0001e+05, 2.0000e+01, 3.0000e+01], dtype=float64)

    """
    del role, kw
    usys = eqx.error_if(usys, usys is None, "usys required")

    # For ArrayLike, we only support Cartesian charts.
    cart = chart.cartesian
    chart = eqx.error_if(
        chart, chart != cart, "Boost(Vel, ArrayLike) requires a Cartesian chart"
    )

    x_arr = jnp.asarray(x)

    # Materialize dv
    op_eval = eval_op(op, tau)

    # Restrict to matching Cartesian charts for ArrayLike.
    op_chart = op.chart
    op_cart = op_chart.cartesian
    _ = eqx.error_if(
        op_chart,
        op_chart != op_cart or op_cart != chart,
        "Boost(Vel, ArrayLike) requires op.chart to match the input "
        "Cartesian chart; use CsDict/Quantity for nontrivial chart conversions.",
    )

    dv_vals, dv_unit = pack_uniform_unit(op_eval.delta, keys=chart.components)

    # Convert to canonical velocity unit in usys, then strip to a raw array.
    vel_unit = usys[u.dimension_of(u.unit("m/s"))]
    if dv_unit is not None:
        dv_vals = u.uconvert_value(vel_unit, dv_unit, dv_vals)

    return x_arr + dv_vals if op_eval.right_add else dv_vals + x_arr


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysAcc,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Boost to an PhysAcc-valued ArrayLike.

    ArrayLike is interpreted as Cartesian acceleration components in the
    canonical acceleration unit from `usys`. For time-dependent `dv`, `tau`
    must be a time Quantity. This dispatch only supports the case `op.chart`
    matches the input Cartesian chart.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u

    Time-independent boost is identity on acceleration:

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> a = jnp.array([1.0, 2.0, 3.0])
    >>> usys = u.unitsystems.si
    >>> cxop.apply_op(boost, None, cxr.phys_acc, cxc.cart3d, a, usys=usys)
    Array([1., 2., 3.], dtype=float64)

    """
    del role, kw

    usys = eqx.error_if(usys, usys is None, "usys required")

    cart = chart.cartesian
    chart = eqx.error_if(
        chart, chart != cart, "Boost(Acc, ArrayLike) requires a Cartesian chart"
    )

    x_arr = jnp.asarray(x)

    # Time-independent boost: identity on acceleration.
    if not callable(op.delta):
        return x_arr

    # Restrict to matching Cartesian charts for ArrayLike.
    op_chart = op.chart
    op_cart = op_chart.cartesian
    _ = eqx.error_if(
        op_chart,
        op_chart != op_cart or op_cart != chart,
        "Boost(Acc, ArrayLike) with time-dependent dv requires op.chart to "
        "match the input Cartesian chart; use CsDict/Quantity for nontrivial "
        "chart conversions.",
    )

    d1_op, _ = _time_derivative_delta(op, tau, order=1)
    da_vals, da_unit = pack_uniform_unit(d1_op, keys=chart.components)

    # Convert to canonical acceleration unit in usys, then strip to a raw array.
    acc_unit = usys[u.dimension_of(u.unit("m/s2"))]
    if da_unit is not None:
        da_vals = u.uconvert_value(acc_unit, da_unit, da_vals)

    return x_arr + da_vals if eval_op(op, tau).right_add else da_vals + x_arr


# ------------------------------
# Quantity dispatches

_MSG_NEEDS_AT_BOOST: Final = (
    "Boost on physical tangent roles requires `at=` (base point) when chart "
    "conversion is needed. Provide `at` as a Point-valued CsDict in the same "
    "`chart` as `x`."
)


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Boost to a Point-valued Quantity.

    Per spec: Boost acts on Point as a time-dependent translation:
    p'(tau) = p(tau) + (tau - tau0) * delta_v

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> x = u.Q([0.0, 0.0, 0.0], "km")
    >>> tau = u.Q(1.0, "s")
    >>> cxop.apply_op(boost, tau, cxr.point, cxc.cart3d, x)
    Quantity(Array([100., 0., 0.], dtype=float64), unit='km')

    """
    del role, kw, usys

    op_eval = eval_op(op, tau)
    dv = jnp.stack([op_eval.delta[k] for k in op.chart.components], axis=-1)

    # Check if dv is nonzero - if so, require tau to be a time Quantity
    if not jnp.allclose(u.ustrip(AllowValue, dv), 0):
        _require_tau_time(tau)  # validate tau is a time Quantity

        # Compute displacement: delta_x = (tau - tau0) * delta_v
        # Assuming tau0 = 0 for simplicity
        # Use tau directly (with time unit) so tau * velocity = length
        displacement = tau * dv

        return x + displacement if op_eval.right_add else displacement + x
    # Zero boost: identity on points
    return x


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysDisp,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Boost to a Pos-valued Quantity.

    Per spec: Galilean boost is identity on PhysDisp.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> d = u.Q([10.0, 20.0, 30.0], "km")
    >>> cxop.apply_op(boost, None, cxr.phys_disp, cxc.cart3d, d)
    Quantity(Array([10., 20., 30.], dtype=float64), unit='km')

    """
    del op, tau, role, chart, usys, kw
    return x


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysVel,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Boost to a Vel-valued Quantity.

    The dv is added to the velocity. For time-dependent dv, it is evaluated
    at tau.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> v = u.Q([10.0, 20.0, 30.0], "km/s")
    >>> cxop.apply_op(boost, None, cxr.phys_vel, cxc.cart3d, v)
    Quantity(Array([110.,  20.,  30.], dtype=float64), unit='km / s')

    """
    del role, kw, usys

    op_eval = eval_op(op, tau)
    dv = jnp.stack([op_eval.delta[k] for k in op.chart.components], axis=-1)

    return x + dv if op_eval.right_add else dv + x


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysAcc,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Boost to an PhysAcc-valued Quantity.

    If `dv` is time-independent, this is a no-op.

    If `dv` is time-dependent, the induced acceleration shift is

        a' = a + d(dv)/dt.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Time-independent boost is a no-op on acceleration:

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> a = u.Q([1.0, 2.0, 3.0], "m/s**2")
    >>> cxop.apply_op(boost, None, cxr.phys_acc, cxc.cart3d, a)
    Quantity(Array([1., 2., 3.], dtype=float64), unit='m / s2')

    """
    del role, kw, usys

    if not callable(op.delta):
        return x

    d1, _ = _time_derivative_delta(op, tau, order=1)
    da = jnp.stack([d1[k] for k in op.chart.components], axis=-1)

    return x + da if eval_op(op, tau).right_add else da + x


# ------------------------------
# CsDict dispatches


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    """Apply Boost to a Point-valued CsDict.

    Per spec: Boost acts on Point as a time-dependent translation:
    p'(tau) = p(tau) + (tau - tau0) * delta_v

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> x = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> tau = u.Q(1.0, "s")
    >>> result = cxop.apply_op(boost, tau, cxr.point, cxc.cart3d, x)
    >>> result["x"]
    Quantity(Array(100., dtype=float64), unit='km')

    """
    del role, kw, usys

    op_eval = eval_op(op, tau)

    # Check if dv is nonzero - if so, require tau to be a time Quantity
    is_zero = jtu.all(
        jtu.map(lambda v: jnp.allclose(u.ustrip(AllowValue, v), 0), op_eval.delta)
    )
    if not is_zero:
        tau_value, tau_unit = _require_tau_time(tau)
        tau_q = u.Q(tau_value, tau_unit)

        # Compute displacement: delta_x = (tau - tau0) * delta_v
        # Assuming tau0 = 0 for simplicity
        displacement = {k: tau_q * op_eval.delta[k] for k in chart.components}

        # Add displacement to point
        if op_eval.right_add:
            return {k: x[k] + displacement[k] for k in chart.components}
        return {k: displacement[k] + x[k] for k in chart.components}
    # Zero boost: identity on points
    return x


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysDisp,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    """Apply Boost to a Pos-valued CsDict.

    Per spec: Galilean boost is identity on Pos (physical displacements).

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> d = {"x": u.Q(10, "km"), "y": u.Q(20, "km"), "z": u.Q(30, "km")}
    >>> result = cxop.apply_op(boost, None, cxr.phys_disp, cxc.cart3d, d)
    >>> result == d
    True

    """
    del op, tau, role, chart, usys, kw
    return x


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysVel,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    r"""Apply Boost to a Vel-valued CsDict.

    The velocity boost $\Delta v$ is added to the velocity. The shift is
    computed in `op.chart` and converted to `chart` as a *physical tangent
    vector* evaluated at the base point `at`.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> v = {"x": u.Q(10, "km/s"), "y": u.Q(20, "km/s"),
    ...      "z": u.Q(30, "km/s")}
    >>> cxop.apply_op(boost, None, cxr.phys_vel, cxc.cart3d, v)
    {'x': Quantity(..., unit='km / s'), ...}

    """
    del role, kw

    op_eval = eval_op(op, tau)

    # Need a base point if any physical tangent transform is required.
    needs_at = (chart != chart.cartesian) or (op.chart != op.chart.cartesian)
    x = eqx.error_if(x, needs_at and at is None, _MSG_NEEDS_AT_BOOST)

    if not needs_at:
        # Both charts are Cartesian; components align by convention.
        dv = op_eval.delta
    else:
        # Express base point in the operator chart for the tangent transform.
        at0 = at if at is not None else {}
        at_in_op = cxapi.point_transform(op.chart, chart, at0, usys=usys)
        dv = cxapi.physical_tangent_transform(
            chart, op.chart, op_eval.delta, at=at_in_op, usys=usys
        )

    return jtu.map(
        jnp.add,
        *((x, dv) if op_eval.right_add else (dv, x)),
        is_leaf=u.quantity.is_any_quantity,
    )


@plum.dispatch
def apply_op(
    op: Boost,
    tau: Any,
    role: cxr.PhysAcc,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    r"""Apply Boost to an PhysAcc-valued CsDict.

    If `dv` is time-independent: no-op.

    If `dv` is time-dependent, the induced acceleration shift is

        a'(\tau) = a(\tau) + d(dv)/dt.

    The shift is computed in `op.chart` and converted to `chart` as a
    *physical tangent vector* evaluated at the base point `at`.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Time-independent boost is a no-op on acceleration:

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> a = {"x": u.Q(1, "m/s**2"), "y": u.Q(2, "m/s**2"),
    ...      "z": u.Q(3, "m/s**2")}
    >>> result = cxop.apply_op(boost, None, cxr.phys_acc, cxc.cart3d, a)
    >>> result == a
    True

    """
    del role, kw

    if not callable(op.delta):
        return x

    needs_at = (chart != chart.cartesian) or (op.chart != op.chart.cartesian)
    x = eqx.error_if(x, needs_at and at is None, _MSG_NEEDS_AT_BOOST)

    d1_op, _ = _time_derivative_delta(op, tau, order=1)

    if not needs_at:
        da = d1_op
    else:
        at0 = at if at is not None else {}
        at_in_op = cxapi.point_transform(op.chart, chart, at0, usys=usys)
        da = cxapi.physical_tangent_transform(
            chart, op.chart, d1_op, at=at_in_op, usys=usys
        )

    return jtu.map(
        jnp.add,
        *((x, da) if eval_op(op, tau).right_add else (da, x)),
        is_leaf=u.quantity.is_any_quantity,
    )
