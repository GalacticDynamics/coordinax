"""Translation operator."""

__all__ = ("Translate",)


from collections.abc import Callable
from jaxtyping import Array, ArrayLike
from typing import Any, Final, Protocol, final, runtime_checkable

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


@runtime_checkable
class CanAddandNeg(Protocol):
    """Protocol for classes that support addition and negation."""

    def __add__(self, other: Any, /) -> Any: ...
    def __neg__(self, /) -> Any: ...


@final
class Translate(AbstractAdd):
    r"""Operator for translating points.

    A Translate operator represents addition of a constant displacement $\Delta$
    in the ambient Euclidean space (or in a chart whose metric is Euclidean and
    whose canonical Cartesian chart exists).

    Think of $\Delta$ as a {class}`~coordinax.roles.PhysDisp`-valued vector field
    that is constant in space and time (unless explicitly time-dependent).

    Formally, in a Cartesian chart on $\mathbb{R}^n$: $T_\Delta:\; x \mapsto
    x+\Delta$.

    Its differential (pushforward) is the identity: $(dT_\Delta)_x = I$.

    Parameters
    ----------
    delta : CsDict | Callable[[tau], CsDict]
        The displacement to apply. Must have length dimension.  If callable,
        will be evaluated at the time parameter ``tau``.

    Notes
    -----
    - Only applicable to ``Point`` role vectors
    - Raises ``TypeError`` when applied to other roles (e.g., ``PhysVel``)
    - The ``delta`` is interpreted as a ``PhysDisp``-role displacement internally

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.ops as cxop
    >>> import wadler_lindig as wl

    Create a translation operator:

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> shift
    Translate({'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}, chart=Cart3D())

    The inverse negates the displacement:

    >>> shift.inverse
    Translate(
        {'x': Q(-1, 'km'), 'y': Q(-2, 'km'), 'z': Q(-3, 'km')}, chart=Cart3D()
    )

    Time-dependent translation:

    >>> delta = lambda t: {"x": u.Q(t.ustrip("s"), "m"), "y": u.Q(0, "m"),
    ...                    "z": u.Q(0, "m")}
    >>> moving = cxop.Translate(delta, chart=cxc.cart3d)
    >>> moving
    Translate(<function <lambda>>, chart=Cart3D())

    >>> t = u.Q(10, "s")
    >>> x = cx.cdict(u.Q([0, 0, 0], "m"))
    >>> wl.pprint(moving(t, x), short_arrays='compact', named_units=False)
    {'x': Quantity(10, unit='m'), 'y': Quantity(0, unit='m'),
     'z': Quantity(0, unit='m')}

    See Also
    --------
    Boost : Velocity offset operator

    """

    # delta, chart, and right_add inherited from AbstractAdd
    # role: Final = cxr.phys_disp

    # inverse, __add__, and __neg__ inherited from AbstractAdd
    # __add__ and __neg__ inherited from AbstractAdd


# ============================================================================
# Constructors


@Translate.from_.dispatch
def from_(cls: type[Translate], obj: Translate, /) -> Translate:
    """Construct a Translate from another Translate.

    Examples
    --------
    >>> import coordinax as cx
    >>> shift1 = cxop.Translate.from_([1, 2, 3], "km")
    >>> cxop.Translate.from_(shift1) is shift1
    True

    """
    return obj


@Translate.from_.dispatch
def from_(cls: type[Translate], q: u.AbstractQuantity, /) -> Translate:
    """Construct a Translate from a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxop

    >>> cxop.Translate.from_(u.Q([1, 2, 3], "km"))
    Translate({'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}, chart=Cart3D())

    """
    chart = cxapi.guess_chart(q)
    x = cxapi.cdict(q, chart)
    return cls(x, chart=chart)


# ============================================================================
# Simplification


@plum.dispatch
def simplify(op: Translate, /, **kw: Any) -> Translate | Identity:
    """Simplify a Translate operator.

    A translation with zero delta simplifies to Identity.

    Examples
    --------
    >>> import coordinax.ops as cxop

    >>> op = cxop.Translate.from_([1, 2, 3], "km")
    >>> cxop.simplify(op)
    Translate(...)

    >>> op = cxop.Translate.from_([0, 0, 0], "km")
    >>> cxop.simplify(op)
    Identity()

    """
    is_zero = jtu.all(
        jtu.map(lambda v: jnp.allclose(u.ustrip(AllowValue, v), 0, **kw), op.delta)
    )
    if is_zero:
        return Identity()  # type: ignore[no-untyped-call]
    return op


# ============================================================================
# apply_op


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Translate to an ArrayLike.

    The array is interpreted as Cartesian coordinates. The delta is converted
    to the same unit system to perform the addition.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> usys = u.unitsystems.si
    >>> cxop.apply_op(shift, None, cxr.point, cxc.cart3d, x, usys=usys)
    Array([1000., 2000., 3000.], dtype=float64)

    """
    del role, kw
    usys = eqx.error_if(usys, usys is None, "requires usys")
    chart = eqx.error_if(
        chart, not isinstance(chart, type(chart.cartesian)), "chart must be cartesian"
    )

    # Process Translation
    op_eval = eval_op(op, tau)

    # Convert delta to array using chart components and usys
    delta, unit = pack_uniform_unit(op_eval.delta, chart.components)
    if unit is not None:
        delta = u.uconvert_value(usys[u.dimension_of(unit)], unit, delta)

    # Apply translation
    x_arr = jnp.asarray(x)
    return x_arr + delta if op_eval.right_add else delta + x_arr


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.PhysDisp,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: Any,
    /,
    **kw: Any,
) -> Array:
    """Return unchanged; translation has derivative identity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> v = jnp.array([10.0, 20.0, 30.0])
    >>> cxop.apply_op(shift, None, cxr.phys_disp, cxc.cart3d, v)
    Array([10., 20., 30.], dtype=float64)

    """
    del op, tau, role, chart, kw
    return x


def _require_tau_time(tau: Any, /) -> tuple[Array, Unit]:
    """Return (tau_value, tau_unit), requiring tau to be a time quantity."""
    # We require a Quantity-like tau with time dimension so we can form d/dtau.
    tau = eqx.error_if(
        tau,
        not u.quantity.is_any_quantity(tau),
        "Translate requires tau as a time Quantity when delta is time-dependent",
    )
    tau_unit = u.unit_of(tau)
    tau = eqx.error_if(
        tau,
        u.dimension_of(tau_unit) != u.dimension_of(u.unit("s")),
        "Translate requires tau to have time dimension",
    )
    # Differentiate with respect to the numeric tau in its own unit.
    return jnp.asarray(u.ustrip(tau_unit, tau)), tau_unit


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
        f: tau_value -> Array[... , n] of delta values in `chart.components` order.
        unit: common length unit returned by `pack_uniform_unit`.

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


def _dt_delta(
    op: Translate, tau: Any, /, *, order: int
) -> tuple[CsDict, cxc.AbstractChart[Any, Any]]:
    """Compute d^order/dtau^order delta(tau) as a CsDict in op.chart.

    Returns
    -------
    (deltas, chart)
        `deltas` has the same component keys as `op.chart.components`.
        Units are `length / time^order`.

    """
    if not callable(op.delta):
        raise TypeError("_dt_delta requires callable delta")

    tau_value, tau_unit = _require_tau_time(tau)

    f, length_unit = _delta_values_fn(op.delta, op.chart, tau_unit)

    if order == 1:
        df = jax.jacfwd(f)(tau_value)
        out_vals = df
        out_unit = None if length_unit is None else (length_unit / tau_unit)
    elif order == 2:
        ddf = jax.jacfwd(jax.jacfwd(f))(tau_value)
        out_vals = ddf
        out_unit = None if length_unit is None else (length_unit / (tau_unit**2))
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


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.PhysVel,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Translate to a Vel-valued ArrayLike.

    ArrayLike is interpreted as Cartesian velocity components in the canonical
    velocity unit from `usys`. For time-dependent `delta`, `tau` must be a time
    Quantity. This dispatch only supports the case `op.chart` matches the input
    Cartesian chart; otherwise users must pass `Quantity` or `CsDict`.
    """
    del role, kw

    # Arrays are interpreted as Cartesian components in the canonical unit for the
    # role's dimension in the provided unit system.
    usys = eqx.error_if(usys, usys is None, "usys required")

    # For ArrayLike, we only support Cartesian charts without needing a base point.
    cart = chart.cartesian
    chart = eqx.error_if(
        chart, chart != cart, "Translate(Vel, ArrayLike) requires a Cartesian chart"
    )

    x_arr = jnp.asarray(x)

    # Time-independent translation: derivative is identity => no-op on velocity.
    if not callable(op.delta):
        return x_arr

    # For time-dependent delta, add d(delta)/dt expressed in usys velocity units.
    # Restrict to the case where the delta is defined in the same Cartesian chart;
    # otherwise a physical tangent transform would require a base point.
    op_chart = op.chart
    op_cart = op_chart.cartesian
    _ = eqx.error_if(
        op_chart,
        op_chart != op_cart or op_cart != chart,
        "Translate(Vel, ArrayLike) with time-dependent delta requires "
        "op.chart to match the input Cartesian chart; use CsDict/Quantity "
        "for nontrivial chart conversions.",
    )

    d1_op, _ = _dt_delta(op, tau, order=1)
    dv_vals, dv_unit = pack_uniform_unit(d1_op, keys=chart.components)

    # Convert to canonical velocity unit in usys, then strip to a raw array.
    vel_unit = usys[u.dimension_of(u.unit("m/s"))]
    if dv_unit is not None:
        dv_vals = u.uconvert_value(vel_unit, dv_unit, dv_vals)

    return x_arr + dv_vals if eval_op(op, tau).right_add else dv_vals + x_arr


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.PhysAcc,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> Array:
    """Apply Translate to an PhysAcc-valued ArrayLike.

    ArrayLike is interpreted as Cartesian acceleration components in the
    canonical acceleration unit from `usys`.  For time-dependent `delta`, `tau`
    must be a time Quantity.  This dispatch only supports the case `op.chart`
    matches the input Cartesian chart; otherwise users must pass `Quantity` or
    `CsDict`.
    """
    del role, kw

    usys = eqx.error_if(usys, usys is None, "usys required")

    cart = chart.cartesian
    chart = eqx.error_if(
        chart, chart != cart, "Translate(Acc, ArrayLike) requires a Cartesian chart"
    )

    x_arr = jnp.asarray(x)

    # Time-independent translation: derivative is identity => no-op on acceleration.
    if not callable(op.delta):
        return x_arr

    # Restrict to matching Cartesian charts; otherwise a base point would be required.
    op_chart = op.chart
    op_cart = op_chart.cartesian
    _ = eqx.error_if(
        op_chart,
        op_chart != op_cart or op_cart != chart,
        "Translate(Acc, ArrayLike) with time-dependent delta requires "
        "op.chart to match the input Cartesian chart; use CsDict/Quantity "
        "for nontrivial chart conversions.",
    )

    d2_op, _ = _dt_delta(op, tau, order=2)
    da_vals, da_unit = pack_uniform_unit(d2_op, keys=chart.components)

    # Convert to canonical acceleration unit in usys, then strip to a raw array.
    acc_unit = usys[u.dimension_of(u.unit("m/s2"))]
    if da_unit is not None:
        da_vals = u.uconvert_value(acc_unit, da_unit, da_vals)

    return x_arr + da_vals if eval_op(op, tau).right_add else da_vals + x_arr


# -----------------------------------------------
# Special dispatches for Quantity.
# These are interpreted as Cartesian coordinates in a Euclidean metric
# The role is inferred from the dimensions.


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Translate to a Quantity.

    The array is interpreted as Cartesian coordinates. The delta is converted
    to the same unit system to perform the addition.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> x = u.Q([0.0, 0.0, 0.0], "m")
    >>> cxop.apply_op(shift, None, cxr.point, cxc.cart3d, x)
    Quantity(Array([1000., 2000., 3000.], dtype=float64), unit='m')

    """
    # Process Translation
    op_eval = eval_op(op, tau)

    # Convert delta to array using chart components and usys
    delta = jnp.stack(list(op_eval.delta.values()), axis=-1)

    return x + delta if op_eval.right_add else delta + x


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.PhysVel,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Translate to a Vel-roled Quantity.

    If `delta` is time-independent, this is a no-op.

    If `delta` is time-dependent, the induced velocity shift is

        v' = v + d(delta)/dt.

    The shift is evaluated at `tau` and returned in the same units as `x`.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Time-independent translation is a no-op on velocity:

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> v = u.Q([10.0, 20.0, 30.0], "km/s")
    >>> cxop.apply_op(shift, None, cxr.phys_vel, cxc.cart3d, v)
    Quantity(Array([10., 20., 30.], dtype=float64), unit='km / s')

    """
    del role, kw, usys

    if not callable(op.delta):
        return x

    # Compute d(delta)/dt in op.chart, then pack in op.chart component order.
    d1, _ = _dt_delta(op, tau, order=1)
    dv = jnp.stack([d1[k] for k in op.chart.components], axis=-1)

    return x + dv if eval_op(op, tau).right_add else dv + x


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.PhysAcc,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Translate to an PhysAcc-roled Quantity.

    If `delta` is time-independent, this is a no-op.

    If `delta` is time-dependent, the induced acceleration shift is

        $$ a' = a + d^2(delta)/dt^2. $$

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Time-independent translation is a no-op on acceleration:

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> a = u.Q([1.0, 2.0, 3.0], "m/s**2")
    >>> cxop.apply_op(shift, None, cxr.phys_acc, cxc.cart3d, a)
    Quantity(Array([1., 2., 3.], dtype=float64), unit='m / s2')

    """
    del role, kw, usys

    if not callable(op.delta):
        return x

    d2, _ = _dt_delta(op, tau, order=2)
    da = jnp.stack([d2[k] for k in op.chart.components], axis=-1)

    return x + da if eval_op(op, tau).right_add else da + x


# -----------------------------------------------
# On CsDict


@plum.dispatch
def apply_op(
    op: Translate,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    """Apply Translate to a Point-valued component dictionary.

    Translation is performed in the canonical Cartesian chart of ``chart``:

    1. Convert the point ``x`` to ``cart = chart.cartesian``.
    2. Convert ``delta`` to the same ``cart`` using
        ``api.physical_tangent_transform`` evaluated at the point being
        translated (expressed in ``op.chart``).
    3. Add in Cartesian components.
    4. Convert the translated point back to ``chart``.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> x = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxop.apply_op(shift, None, cxr.point, cxc.cart3d, x)
    {'x': Quantity(Array(1, dtype=int64), unit='km'),
     'y': Quantity(Array(2, dtype=int64), unit='km'),
     'z': Quantity(Array(3, dtype=int64), unit='km')}

    """
    del role, kw

    op_eval = eval_op(op, tau)

    # Shared canonical Cartesian chart for performing the translation.
    cart = chart.cartesian

    # Convert point to Cartesian.
    x_cart = cxapi.point_transform(cart, chart, x, usys=usys)

    # Convert delta to Cartesian (as a physical displacement).
    if op_eval.chart == cart:
        delta_cart = op_eval.delta
    else:
        # Base point expressed in the delta's chart.
        at_in_op_chart = cxapi.point_transform(op_eval.chart, chart, x, usys=usys)
        delta_cart = cxapi.physical_tangent_transform(
            cart, op_eval.chart, op_eval.delta, at=at_in_op_chart, usys=usys
        )

    # Add in Cartesian components (key-wise).
    x_cart2 = jtu.map(
        jnp.add,
        *((x_cart, delta_cart) if op_eval.right_add else (delta_cart, x_cart)),
        is_leaf=u.quantity.is_any_quantity,
    )

    # Convert back to the input chart.
    return cxapi.point_transform(chart, cart, x_cart2, usys=usys)


_MSG_NEEDS_AT_TRANSLATE: Final = (
    "Translate on physical tangent roles requires `at=` (base point) when chart "
    "conversion is needed. Provide `at` as a Point-valued CsDict in the same "
    "`chart` as `x`."
)


@plum.dispatch
def apply_op(
    op: Translate,
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
    r"""Apply Translate to a Vel-valued component dictionary.

    If `delta` is time-independent: no-op.

    If `delta` is time-dependent, the induced velocity shift is

    $$ v'(\tau) = v(\tau) + \dot\Delta(\tau). $$

    The shift is computed in `op.chart` and converted to `chart` as a *physical
    tangent vector* evaluated at the base point `at`.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Time-independent translation is a no-op on velocity:

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> v = {"x": u.Q(10, "km/s"), "y": u.Q(20, "km/s"), "z": u.Q(30, "km/s")}
    >>> cxop.apply_op(shift, None, cxr.phys_vel, cxc.cart3d, v)
    {'x': Quantity(..., unit='km / s'), 'y': Quantity(...), 'z': Quantity(...)}

    """
    del role, kw

    if not callable(op.delta):
        return x

    # Need a base point if any physical tangent transform is required.
    needs_at = (chart != chart.cartesian) or (op.chart != op.chart.cartesian)
    x = eqx.error_if(x, needs_at and at is None, _MSG_NEEDS_AT_TRANSLATE)

    d1_op, _ = _dt_delta(op, tau, order=1)

    if not needs_at:
        # Both charts are Cartesian; components align by convention.
        dv = d1_op
    else:
        # Express base point in the operator chart for the tangent transform.
        at0 = at if at is not None else {}
        at_in_op = cxapi.point_transform(op.chart, chart, at0, usys=usys)
        dv = cxapi.physical_tangent_transform(
            chart, op.chart, d1_op, at=at_in_op, usys=usys
        )

    return jtu.map(
        jnp.add,
        *((x, dv) if eval_op(op, tau).right_add else (dv, x)),
        is_leaf=u.quantity.is_any_quantity,
    )


@plum.dispatch
def apply_op(
    op: Translate,
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
    r"""Apply Translate to an PhysAcc-valued component dictionary.

    If `delta` is time-independent: no-op.

    If `delta` is time-dependent, the induced acceleration shift is

    $$ a'(\tau) = a(\tau) + \ddot\Delta(\tau). $$

    The shift is computed in `op.chart` and converted to `chart` as a *physical
    tangent vector* evaluated at the base point `at`.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Time-independent translation is a no-op on acceleration:

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> a = {"x": u.Q(1, "m/s**2"), "y": u.Q(2, "m/s**2"), "z": u.Q(3, "m/s**2")}
    >>> cxop.apply_op(shift, None, cxr.phys_acc, cxc.cart3d, a)
    {'x': Quantity(..., unit='m / s2'), 'y': Quantity(...), 'z': Quantity(...)}

    """
    del role, kw

    if not callable(op.delta):
        return x

    needs_at = (chart != chart.cartesian) or (op.chart != op.chart.cartesian)
    x = eqx.error_if(x, needs_at and at is None, _MSG_NEEDS_AT_TRANSLATE)

    d2_op, _ = _dt_delta(op, tau, order=2)

    if not needs_at:
        da = d2_op
    else:
        at0 = at if at is not None else {}
        at_in_op = cxapi.point_transform(op.chart, chart, at0, usys=usys)
        da = cxapi.physical_tangent_transform(
            chart, op.chart, d2_op, at=at_in_op, usys=usys
        )

    return jtu.map(
        jnp.add,
        *((x, da) if eval_op(op, tau).right_add else (da, x)),
        is_leaf=u.quantity.is_any_quantity,
    )
