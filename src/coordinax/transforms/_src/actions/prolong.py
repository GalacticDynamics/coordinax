r"""Generic jet-prolongation engine for transform actions.

This module implements the *kinematic prolongation* of a transform's point
action to tangent data of any time-derivative order, via forward-mode
automatic differentiation.

Mathematical background:

Let $\phi(\tau, x)$ be the point action of a transform (possibly with
time-dependent parameters). If a curve satisfies $x'(\tau) = \phi(\tau,
x(\tau))$ then the transformed velocity and acceleration are the total
$\tau$-derivatives

$$
v' = \partial_\tau \phi + \partial_x \phi \cdot v, \qquad
a' = \partial_{\tau\tau} \phi + 2 \partial_\tau \partial_x \phi \cdot v
     + \partial_{xx} \phi(v, v) + \partial_x \phi \cdot a,
$$

and so on for higher orders. These are computed here with nested `jax.jvp`
in the joint $(\tau, x)$ argument, so arbitrary time dependence and
compositions are handled correctly by the chain rule.

Two related verbs are distinguished:

- ``act`` on order-$m$ tangent data ($m \geq 1$: velocity, acceleration, ...)
  is the $m$-th prolongation above.
- ``pushforward`` is the frozen-$\tau$ spatial differential $\partial_x
  \phi(\tau, \cdot) \cdot v$. This is the transformation law for
  `Displacement` data (order 0), which is a same-$\tau$ point difference and
  never gains $\partial_\tau$ terms. For time-independent transforms the two
  verbs coincide.
"""

__all__ = (
    "JetDict",
    "prolong_jet",
    "prolong_slot",
    "pushforward_generic",
    "tau_derivative",
)

from fractions import Fraction

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import jax
import jax.numpy as jax_np
import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity

import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
from .base import AbstractTransform, is_time_dependent
from .custom_types import CDict, OptUSys

JetDict: TypeAlias = dict[int, CDict]
"""A jet of curve data: ``{0: q, 1: v, 2: a, ...}``.

Keys are time-derivative orders of the curve $x(\tau)$: slot 0 is the base
point (position), slot 1 the velocity, slot 2 the acceleration, and so on.
All-integer keys keep the jet a valid JAX pytree (jit/vmap-safe).

Note the distinction from the tangent semantic-kind ladder: `Displacement`
(ladder order 0) is a same-$\tau$ point difference, *not* a curve derivative,
so it is never a jet slot — displacement fibres transform by ``pushforward``.
"""

_MSG_TAU_REQUIRED = (
    "act/prolong for the time-dependent transform {op} on tangent data "
    "requires a time parameter; got tau=None."
)
_MSG_AT_REQUIRED = (
    "{verb}({op}, ...) on order-{m} tangent data requires the base point "
    "via the keyword argument 'at' (a CDict in the same chart), or use a "
    "coordinax.Coordinate bundle which supplies it automatically."
)
_MSG_AT_VEL_REQUIRED = (
    "act({op}, ...) on order-{m} tangent data with a time-dependent "
    "transform requires the lower-order jet slots; pass 'at_vel' (the "
    "velocity at the base point) or use a coordinax.Coordinate bundle."
)
_MSG_JET_SLOT_MISSING = (
    "prolong({op}, ...) requires all jet slots 1..{m}; slot {k} is missing."
)


# =============================================================================
# Unit-handling helpers


def _strip_leaf(unit: Any, leaf: Any, /) -> Any:
    """Strip a leaf to a raw array in the given unit (no-op if unit is None)."""
    return u.ustrip(unit, leaf) if unit is not None else jnp.asarray(leaf)


def _attach_leaf(unit: Any, val: Any, /) -> Any:
    """Attach a unit to a raw value (no-op if unit is None)."""
    return u.Q(val, unit) if unit is not None else val


def _cdict_units(x: CDict, /) -> dict[str, Any]:
    """Per-component units of a CDict (None for raw-array components)."""
    return {k: u.unit_of(v) for k, v in x.items()}


def _strip_cdict(x: CDict, units: dict[str, Any], /) -> dict[str, Any]:
    return {k: _strip_leaf(units[k], v) for k, v in x.items()}


def _attach_cdict(vals: dict[str, Any], units: dict[str, Any], /) -> CDict:
    return {k: _attach_leaf(units[k], v) for k, v in vals.items()}


def _per_time(unit: Any, tau_unit: Any, order: int, /) -> Any:
    """Return the unit ``unit / tau_unit**order`` (None-propagating)."""
    if unit is None:
        return None
    if order == 0 or tau_unit is None:
        return unit
    return unit / tau_unit**order


def _tau_value_unit(tau: Any, /) -> tuple[Any, Any]:
    """Split tau into (raw value, unit-or-None)."""
    tau_unit = u.unit_of(tau)
    tau_val = u.ustrip(tau_unit, tau) if tau_unit is not None else jnp.asarray(tau)
    return tau_val, tau_unit


# =============================================================================
# tau_derivative: unit-aware d^n/dtau^n of a callable parameter


def tau_derivative(f: Callable[[Any], Any], tau: Any, /, *, n: int = 1) -> Any:
    r"""Compute the ``n``-th derivative of ``f`` with respect to ``tau``.

    ``f`` is a callable of a single (possibly unitful) time parameter,
    returning a pytree whose leaves are `unxt.Quantity` or raw arrays (e.g. a
    ``CDict`` transform parameter, or a rotation-matrix array). The result has
    the same tree structure with each leaf's unit divided by
    ``unit(tau)**n``.

    This is the pytree, multi-unit analog of `unxt.experimental.jacfwd`: the
    output units are recorded from one structural evaluation, the computation
    runs on stripped raw values through nested `jax.jvp`, and the units are
    re-attached afterwards.

    Notes
    -----
    - ``f`` must be JAX-traceable.
    - Raw-array leaves stay raw: their derivative values are per
      ``unit(tau)``; callers using unitless parameters are responsible for
      consistent time units (e.g. via a unit system).
    - Batched ``tau`` is supported elementwise (the derivative is taken along
      the all-ones tangent, which is the elementwise derivative for
      broadcasting parameter functions).

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.transforms import tau_derivative

    >>> delta = lambda t: {"x": u.Q(3.0, "km/s") * t, "y": u.Q(0.0, "km")}
    >>> tau_derivative(delta, u.Q(5.0, "s"))
    {'x': Q(3., 'km / s'), 'y': Q(0., 'km / s')}

    Second derivative of a quadratic:

    >>> delta = lambda t: {"x": u.Q(0.5, "m/s2") * t**2}
    >>> tau_derivative(delta, u.Q(4.0, "s"), n=2)
    {'x': Q(1., 'm / s2')}

    """
    if n < 0:
        msg = f"tau_derivative requires n >= 0, got {n}."
        raise ValueError(msg)
    if n == 0:
        return f(tau)

    tau_val, tau_unit = _tau_value_unit(tau)

    # Discover the output structure and per-leaf units without a full
    # evaluation where possible (eval_shape preserves static unit metadata).
    try:
        y0 = jax.eval_shape(f, tau)
    except TypeError:
        # Not abstractly traceable (e.g. concretization inside f): fall back
        # to a real call. Other errors (units, shapes) propagate.
        y0 = f(tau)
    leaves0, treedef = jtu.flatten(y0, is_leaf=is_any_quantity)
    units = [u.unit_of(leaf) for leaf in leaves0]

    def g(tv: Any, /) -> list[Any]:
        t = _attach_leaf(tau_unit, tv)
        leaves, _ = jtu.flatten(f(t), is_leaf=is_any_quantity)
        return [_strip_leaf(un, leaf) for un, leaf in zip(units, leaves, strict=True)]

    gn = g
    for _ in range(n):

        def gnext(tv: Any, /, *, _prev: Callable[[Any], list[Any]] = gn) -> list[Any]:
            return jax.jvp(_prev, (tv,), (jax_np.ones_like(tv),))[1]

        gn = gnext

    out_vals = gn(tau_val)
    out_leaves = [
        _attach_leaf(_per_time(un, tau_unit, n), v)
        for un, v in zip(units, out_vals, strict=True)
    ]
    return jtu.unflatten(treedef, out_leaves)


# =============================================================================
# Generic pushforward: frozen-tau spatial differential


def pushforward_generic(
    op: AbstractTransform,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Frozen-$\tau$ differential $\partial_x \phi(\tau, \cdot) \cdot v$.

    Computes the spatial pushforward of the tangent vector ``v`` (anchored at
    the base point ``at``) under the transform's point action, holding the
    time parameter fixed. This is the transformation law for `Displacement`
    data and, for time-independent transforms, for all tangent kinds.

    """
    order = rep.semantic_kind.order
    if at is None:
        msg = _MSG_AT_REQUIRED.format(verb="pushforward", op=type(op).__name__, m=order)
        raise TypeError(msg)
    if set(v) != set(at):
        missing = sorted(set(at) - set(v))
        extra = sorted(set(v) - set(at))
        msg = (
            f"pushforward({type(op).__name__}, ...): the tangent components "
            f"do not match the base point's {sorted(at)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
            + "."
        )
        raise TypeError(msg)

    in_units = _cdict_units(at)
    out_units = _point_act_units(op, tau, at, chart, usys=usys)

    def f(xv: dict[str, Any], /) -> dict[str, Any]:
        x = _attach_cdict(xv, in_units)
        y = cast("CDict", cxfmapi.act(op, tau, x, chart, cxr.point, usys=usys))
        return _strip_cdict(y, out_units)

    # Strip v consistently with the base point: v_k is expressed in
    # in_unit_k / T for a common time unit T, so the stripped values form a
    # valid coordinate tangent.
    time_unit = _common_time_unit(tau, in_units, _cdict_units(v), order)
    v_vals = {k: _strip_leaf(_per_time(in_units[k], time_unit, order), v[k]) for k in v}
    at_vals = _strip_cdict(at, in_units)

    _, dy = jax.jvp(f, (at_vals,), (v_vals,))
    return _attach_cdict(
        dy, {k: _per_time(un, time_unit, order) for k, un in out_units.items()}
    )


def _point_act_units(
    op: AbstractTransform,
    tau: Any,
    q0: CDict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys,
) -> dict[str, Any]:
    """Per-component output units of the point action, without computing values.

    Units are static metadata on Quantities, so `jax.eval_shape` discovers
    them without evaluating the action; a real evaluation is the fallback for
    non-traceable actions.
    """
    try:
        y0 = jax.eval_shape(
            lambda q: cxfmapi.act(op, tau, q, chart, cxr.point, usys=usys), q0
        )
    except TypeError:
        # Not abstractly traceable (e.g. concretization inside the point
        # action): fall back to a real call. Other errors (units, shapes)
        # propagate.
        y0 = cxfmapi.act(op, tau, q0, chart, cxr.point, usys=usys)
    return _cdict_units(cast("CDict", y0))


def _common_time_unit(
    tau: Any, in_units: dict[str, Any], v_units: dict[str, Any], order: int, /
) -> Any:
    """Choose a common time unit T for stripping order-``order`` tangent data.

    Any dimensionally-consistent T is mathematically valid, but the choice
    fixes the *units* of the output (``out_unit / T**order``). Prefer deriving
    T from the data itself (``T**order = in_unit / v_unit``) so that outputs
    preserve the data's own units — e.g. a ``kpc/Myr`` velocity pushes forward
    to ``kpc/Myr``, not ``kpc/s``. Fall back to ``tau``'s unit, then to
    seconds for a unitful/raw mix.
    """
    if order == 0:
        return None
    for k, vu in v_units.items():
        iu = in_units.get(k)
        if vu is not None and iu is not None:
            ratio = iu / vu
            return ratio if order == 1 else ratio ** Fraction(1, order)
    tau_unit = u.unit_of(tau) if tau is not None else None
    if tau_unit is not None:
        return tau_unit
    if any(un is not None for un in v_units.values()):
        return u.unit("s")
    return None


def _chain_time(tau: Any, time_unit: Any, /) -> tuple[Any, Callable[[Any], Any]]:
    """Prepare tau for the derivative chain.

    Returns the raw chain value of tau (expressed per the common time unit T)
    and a converter mapping chain values back to what the point action
    expects:

    - ``tau=None``: the point action is applied AT tau=None — no dummy time
      is fabricated; the (unused) chain slot contributes exactly zero, and a
      point action that genuinely requires a time raises its own informative
      error instead of being silently evaluated at a made-up instant.
    - unitful tau: expressed in T so the chain rule's dtau- and
      dx-contributions add consistently.
    - raw (unitless) tau: passed through raw — the user's callables expect a
      number — with the derivative direction interpreted as per T (the
      documented raw-tau convention; cf. `tau_derivative`).
    """
    tau_is_unitful = tau is not None and u.unit_of(tau) is not None
    if tau is None:
        tau_val: Any = jnp.zeros(())  # unused: to_time ignores its argument
    elif tau_is_unitful and time_unit is not None:
        tau_val = u.ustrip(time_unit, tau)
    else:
        tau_val = jnp.asarray(tau)

    def to_time(tv: Any, /) -> Any:
        if tau is None:
            return None
        return _attach_leaf(time_unit, tv) if tau_is_unitful else tv

    return tau_val, to_time


# =============================================================================
# prolong_jet: joint kinematic prolongation of a jet of coordinate data


def prolong_jet(
    op: AbstractTransform,
    tau: Any,
    jet: JetDict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> JetDict:
    r"""Apply the kinematic prolongation of ``op`` to a jet of curve data.

    ``jet`` maps time-derivative orders of the curve to component
    dictionaries: ``{0: q, 1: v, 2: a, ...}``. Slot 0 (the base point) is
    transformed by the point action; each slot $m \geq 1$ by the $m$-th total
    $\tau$-derivative of the point action along the curve the jet represents
    (nested `jax.jvp`).

    All slots ``0..max(jet)`` must be present: the $m$-th prolongation
    depends on every lower-order slot.

    Examples
    --------
    A uniformly moving translation acting on a phase-space jet — the velocity
    gains the translation's rate:

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm

    >>> delta = lambda t: {"x": u.Q(3.0, "km/s") * t, "y": u.Q(0.0, "km"),
    ...                    "z": u.Q(0.0, "km")}
    >>> op = cxfm.Translate(delta, chart=cxc.cart3d)
    >>> jet = {
    ...     0: {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")},
    ...     1: {"x": u.Q(0.0, "km/s"), "y": u.Q(1.0, "km/s"), "z": u.Q(0.0, "km/s")},
    ... }
    >>> out = cxfm.prolong(op, u.Q(2.0, "s"), jet, cxc.cart3d)
    >>> out[0]["x"], out[1]["x"]
    (Q(7., 'km'), Q(3., 'km / s'))

    """
    if 0 not in jet:
        msg = "prolong requires the base point at jet slot 0."
        raise TypeError(msg)
    q0 = jet[0]
    max_order = max(jet)
    for m in range(1, max_order + 1):
        if m not in jet:
            msg = _MSG_JET_SLOT_MISSING.format(op=type(op).__name__, m=max_order, k=m)
            raise TypeError(msg)
        if set(jet[m]) != set(q0):
            missing = sorted(set(q0) - set(jet[m]))
            extra = sorted(set(jet[m]) - set(q0))
            msg = (
                f"prolong({type(op).__name__}, ...): jet slot {m} components "
                f"do not match slot 0's {sorted(q0)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
                + "."
            )
            raise TypeError(msg)

    if is_time_dependent(op) and tau is None and max_order >= 1:
        msg = _MSG_TAU_REQUIRED.format(op=type(op).__name__)
        raise TypeError(msg)

    if max_order == 0:
        y0 = cast("CDict", cxfmapi.act(op, tau, q0, chart, cxr.point, usys=usys))
        return {0: y0}

    in_units = _cdict_units(q0)
    out_units = _point_act_units(op, tau, q0, chart, usys=usys)
    # The common time unit T fixes the units of the output slots
    # (out_unit / T**m); derive it from the data so units are preserved.
    # Everything in the chain — the jet slots AND tau itself — is expressed
    # per T so the chain rule's dtau- and dx-contributions add consistently.
    time_unit = _common_time_unit(tau, in_units, _cdict_units(jet[1]), 1)
    tau_val, to_time = _chain_time(tau, time_unit)

    comps = tuple(q0.keys())

    def f(tv: Any, xv: dict[str, Any], /) -> dict[str, Any]:
        x = _attach_cdict(xv, in_units)
        y = cxfmapi.act(op, to_time(tv), x, chart, cxr.point, usys=usys)
        return _strip_cdict(y, out_units)

    # Stripped jet slots: slot m in units in_unit_k / T**m.
    q0_vals = _strip_cdict(q0, in_units)
    slot_vals = [
        {k: _strip_leaf(_per_time(in_units[k], time_unit, m), jet[m][k]) for k in comps}
        for m in range(1, max_order + 1)
    ]

    slot_outs = _total_derivative_chain(f, tau_val, q0_vals, slot_vals)
    return {
        m: _attach_cdict(
            ym, {k: _per_time(un, time_unit, m) for k, un in out_units.items()}
        )
        for m, ym in enumerate(slot_outs)
    }


def _total_derivative_chain(
    f: Callable[..., dict[str, Any]],
    tau_val: Any,
    q0_vals: dict[str, Any],
    slot_vals: list[dict[str, Any]],
    /,
) -> tuple[dict[str, Any], ...]:
    r"""Evaluate the chain of total-derivative laws of ``f`` along a jet.

    Nests $F_m(t, x, d_1, \ldots, d_m) = \mathrm{jvp}(F_{m-1}, (t, x, d_1,
    \ldots, d_{m-1}), (1, d_1, d_2, \ldots, d_m))$, threading the primals
    through each level so that one evaluation of the outermost $F_M$ yields
    every jet slot $(y_0, y_1, \ldots, y_M)$ — lower orders are not
    recomputed per slot.
    """

    def chain0(t: Any, x: dict[str, Any], /) -> tuple[dict[str, Any], ...]:
        return (f(t, x),)

    chain = chain0
    for _ in slot_vals:

        def chain(
            *a: Any, _prev: Callable[..., tuple[dict[str, Any], ...]] = chain
        ) -> tuple[dict[str, Any], ...]:
            primals, tangents = a[:-1], (jax_np.ones_like(a[0]), *a[2:])
            p, t = jax.jvp(_prev, primals, tangents)
            # p holds (y0..y_{m-1}); the last tangent is d/dtau y_{m-1} = y_m.
            return (*p, t[-1])

    return chain(tau_val, q0_vals, *slot_vals)


# =============================================================================
# Generic plum registrations
#
# All generic (AbstractTransform) rules register at precedence=-1 so that any
# concrete per-operator registration (default precedence 0) wins, as do the
# Identity catch-all (precedence 1) and QMatrix funnels (precedence 2).


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def act(
    op: AbstractTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> CDict:
    """Redispatch on the representation's geometry kind (generic funnel)."""
    return cast("CDict", cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, **kw))


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def act(
    op: AbstractTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.PointGeometry,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> CDict:
    """Raise: a transform must register its own point action (the primitive)."""
    msg = (
        f"{type(op).__name__} does not register a point action "
        "act(op, tau, x: CDict, chart, rep) — the point action is the "
        "primitive every transform must implement."
    )
    raise NotImplementedError(msg)


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def act(
    op: AbstractTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.TangentGeometry,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    at_vel: CDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    r"""Transform tangent data: pushforward or kinematic prolongation.

    - Order-0 (displacement) data and all data under time-independent
      transforms transform by the frozen-$\tau$ ``pushforward``.
    - Under a time-dependent transform, order-$m$ data ($m \geq 1$) transforms
      by the $m$-th prolongation, which requires the lower jet slots: the base
      point ``at`` and, for $m = 2$, the velocity ``at_vel``.

    """
    del kw
    m = rep.semantic_kind.order

    if m == 0 or not is_time_dependent(op):
        return cast(
            "CDict", cxfmapi.pushforward(op, tau, x, chart, rep, at=at, usys=usys)
        )

    return prolong_slot(op, tau, x, chart, m, at=at, at_vel=at_vel, usys=usys)


def prolong_slot(
    op: AbstractTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    m: int,
    /,
    *,
    at: CDict | None,
    at_vel: CDict | None,
    usys: OptUSys,
) -> CDict:
    """Apply the m-th prolongation to a single order-``m`` slot.

    Validates that the lower jet slots are available (the base point ``at``
    and, for ``m == 2``, the velocity ``at_vel``), assembles the jet, and
    returns the transformed slot. Shared by the generic tangent rule and by
    operator fast paths that fall back to the generic prolongation.
    """
    if tau is None:
        raise TypeError(_MSG_TAU_REQUIRED.format(op=type(op).__name__))
    if at is None:
        raise TypeError(_MSG_AT_REQUIRED.format(verb="act", op=type(op).__name__, m=m))

    jet: JetDict = {0: at, m: x}
    if m >= 2:
        if at_vel is None:
            raise TypeError(_MSG_AT_VEL_REQUIRED.format(op=type(op).__name__, m=m))
        jet[1] = at_vel
    if m > 2:
        msg = (
            f"act on order-{m} tangent data requires jet slots 1..{m - 1}; "
            "use coordinax.transforms.prolong with a full jet instead."
        )
        raise TypeError(msg)

    out = cast("JetDict", cxfmapi.prolong(op, tau, jet, chart, usys=usys))
    return out[m]


# -----------------------------------------------------------------------------
# pushforward


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def pushforward(
    op: AbstractTransform,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    """Push tangent data forward via forward-mode AD of the point action."""
    return pushforward_generic(op, tau, v, chart, rep, at=at, usys=usys)


# -----------------------------------------------------------------------------
# prolong


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def prolong(
    op: AbstractTransform,
    tau: Any,
    jet: dict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> dict:
    """Prolong a jet via nested forward-mode AD (generic rule)."""
    return prolong_jet(op, tau, jet, chart, usys=usys)
