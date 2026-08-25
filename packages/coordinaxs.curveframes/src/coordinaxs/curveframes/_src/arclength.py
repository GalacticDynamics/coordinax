r"""Arc-length reparametrisation of a curve.

This module provides `ArcLength`, which wraps a curve
$\boldsymbol{\gamma}(\tau)$ and returns a new curve
$\tilde{\boldsymbol{\gamma}}(s) = \boldsymbol{\gamma}(\tau(s))$ parameterised
by arc length $s$ rather than $\tau$, i.e. $\|\tilde{\boldsymbol{\gamma}}'(s)\|
= 1$ everywhere.

Rather than integrating $s(\tau) = \int \|\boldsymbol{\gamma}'\|\,d\tau$ and then
inverting it, `ArcLength` solves the single ODE

$$ \frac{d\tau}{ds} = \frac{1}{\|\boldsymbol{\gamma}'(\tau)\|},
\qquad \tau(0) = \tau_0, $$

directly for $\tau(s)$.  Because a curve is consumed purely as a callable
throughout `coordinaxs.curveframes` (`AbstractCurveFrameBuilder.__call__` and
`.location` are the only two call sites), `ArcLength(curve, "s")` is itself a
curve and can be wrapped in any of the existing frame builders with no further
change: e.g. ``BishopBuilder(ArcLength(curve, "s"), "km")``. The builder's unit
is a *length*, because what the wrapper exposes is arc length.

"""

__all__ = ("ArcLength", "LagrangianArcLength")

import inspect

from collections.abc import Callable
from typing import Any, ClassVar, cast, final

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffraxtra import DiffEqSolver

import unxt as u

from .attime import AtTime

#: Default integrator for the arc-length ODE.  Same choice as
#: `coordinaxs.curveframes._src.bishop._DIFFEQSOLVER`: `Tsit5` with a tight
#: `PIDController` and `DirectAdjoint`, the only adjoint that is
#: differentiable in both forward and reverse mode -- see that module's
#: comment for the full rationale.
#:
#: `DirectAdjoint` is load-bearing on the *primal* solve too, not only the
#: JVP's quadrature: `_tau_of_s`'s custom rule does not keep autodiff out,
#: because the induced-metric and chart round-trip paths differentiate
#: through the chart in forward mode and so reach `_solve_tau`. Swapping in
#: `RecursiveCheckpointAdjoint` there fails 10 tests with ``can't apply
#: forward-mode autodiff (jvp) to a custom_vjp function``, and measured no
#: faster anyway.
_DIFFEQSOLVER = DiffEqSolver(
    solver=dfx.Tsit5(),
    stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10),
    adjoint=dfx.DirectAdjoint(),
    max_steps=16384,
)


def _speed(
    curve: Callable[[Any], Any],
    tau_unit: u.AbstractUnit,
    s_unit: u.AbstractUnit,
    tau_q: u.AbstractQuantity,
    /,
) -> Any:
    r"""Local speed $\|\boldsymbol{\gamma}'(\tau)\|$ of ``curve`` at ``tau_q``."""
    dcurve = u.experimental.jacfwd(curve, units=(tau_unit,))
    return jnp.linalg.norm(dcurve(tau_q).ustrip(s_unit / tau_unit))


def _rhs_dtau(x: Any, tau_flat: Any, args: Any) -> Any:
    r"""$d\tau/dx = \text{scale}/\|\gamma'(\tau)\|$.

    ``scale`` is what makes this serve both solves: ``s`` when the integration
    variable is the rescaled $\sigma \in [0, 1]$ (`_solve_tau`), and ``1.0``
    when it is $s$ itself (`_solve_tau_dense`).
    """
    del x
    curve, scale, tau_unit, s_unit = args
    return scale / _speed(curve, tau_unit, s_unit, u.Q(tau_flat, tau_unit))


def _rhs_quad(sigma: Any, s_flat: Any, args: Any) -> Any:
    del s_flat
    curve, tau_0_val, dtau, tau_unit, s_unit = args
    tau_q = u.Q(tau_0_val + sigma * dtau, tau_unit)
    return dtau * _speed(curve, tau_unit, s_unit, tau_q)


#: The ODE terms, built **once** at import. Everything that varies per call
#: rides in through `diffrax`'s ``args`` instead of a closure: this is a
#: property of `diffrax.diffeqsolve` itself, which `diffraxtra.DiffEqSolver`
#: inherits by wrapping it -- both are filter-jitted, which puts the term in
#: the static half of the cache key, and a closure hashes by identity. A term
#: rebuilt per call therefore misses the cache and **recompiles the whole
#: integrator every time**: measured at ~350 ms through bare `diffrax` and
#: ~415 ms through `diffraxtra`, against ~0.3 ms of actual integration.
_TERM_DTAU = dfx.ODETerm(_rhs_dtau)
_TERM_QUAD = dfx.ODETerm(_rhs_quad)


def _solve_tau(
    curve: Callable[[Any], Any],
    tau_unit: u.AbstractUnit,
    tau_0: u.AbstractQuantity,
    diffeqsolver: DiffEqSolver,
    s: u.AbstractQuantity,
    /,
) -> u.AbstractQuantity:
    r"""Solve $d\tau/ds = 1/\|\text{curve}'(\tau)\|$ from $\tau_0$ to arc length ``s``.

    Shared by `ArcLength` and `LagrangianArcLength`: both solve the same ODE
    and differ only in which one-argument ``curve`` supplies the speed (the
    slice being evaluated, versus a fixed reference slice) and in which curve
    the resulting $\tau$ is ultimately plugged into.

    ``curve`` must already be one-argument -- a time-dependent curve is bound
    with `AtTime` by the caller before reaching here.
    """
    s_unit = s.unit
    args = (curve, s.ustrip(s_unit), tau_unit, s_unit)
    sol = diffeqsolver(_TERM_DTAU, 0.0, 1.0, None, tau_0.ustrip(tau_unit), args)
    return u.Q(sol.ys[-1], tau_unit)


def _solve_tau_dense(
    curve: Callable[[Any], Any],
    tau_unit: u.AbstractUnit,
    tau_0: u.AbstractQuantity,
    diffeqsolver: DiffEqSolver,
    s_max: u.AbstractQuantity,
    /,
) -> dfx.DenseInterpolation:
    r"""Solve the reparametrisation ODE once, densely, over $[-m, s_{\max} + m]$.

    Companion to `_solve_tau`: the same ODE and right-hand side, but solved a
    single time with `diffrax`'s dense output (``SaveAt(dense=True)``) rather
    than once per requested ``s`` -- what makes the *forward* evaluation cheap
    under repeated calls at a fixed curve (see `ArcLength.s_max`). The margin
    $m$ and why it is integrated rather than clamped: see `_S_MAX_MARGIN`.

    Two solves rather than one: $\tau(0) = \tau_0$ sits in the *middle* of the
    target range, so a short backward solve to $s = -m$ supplies the starting
    value for the forward one.

    No $\sigma \in [0, 1]$ rescaling, unlike `_solve_tau`: that exists to keep
    the derivative well-defined when the differentiated ``s`` is also the
    integration bound, and here the bound is the fixed ``s_max``. The result
    is only ever consumed through `_tau_of_s`'s custom JVP anyway.
    """
    s_unit = s_max.unit
    tau_0_val = tau_0.ustrip(tau_unit)
    s_max_val = s_max.ustrip(s_unit)

    # scale = 1.0: here the integration variable is `s` itself, not a rescaled
    # sigma, so d(tau)/ds is just the reciprocal speed.
    args = (curve, 1.0, tau_unit, s_unit)
    margin = _S_MAX_MARGIN * jnp.abs(s_max_val)

    # Backward from the known tau(0) to the low edge, then forward across the
    # whole extended range from there.
    tau_lo = diffeqsolver(_TERM_DTAU, 0.0, -margin, None, tau_0_val, args).ys[-1]
    sol = diffeqsolver(
        _TERM_DTAU,
        -margin,
        s_max_val + margin,
        None,
        tau_lo,
        args,
        saveat=dfx.SaveAt(dense=True),
    )
    return cast("dfx.DenseInterpolation", sol.interpolation)


#: Fraction of `s_max` that `_solve_tau_dense` integrates *past* each end, so
#: that queries just outside the nominal domain are answered from real solved
#: data rather than clamped to the boundary. Sized to `nearest_tau`'s
#: structural slack: its root-find probes one scan-seed spacing outside
#: `tau_bounds` whenever a seed lands at a domain edge -- measured at exactly
#: `s_max / (n_seed - 1)`, symmetric, so ~1.6% for the default `n_seed=64`.
#: 5% covers that down to `n_seed ~= 21`; below it, widen `s_max`.
_S_MAX_MARGIN = 0.05

_MSG_S_OUT_OF_DOMAIN = (
    "s lies outside the solved domain [0, s_max], beyond the margin "
    "_solve_tau_dense integrates past each end. The interpolation has no "
    "coefficients there, and extrapolating off the end of it would return a "
    "plausible-looking wrong answer. Increase s_max, or leave it `None` to "
    "fall back to solving the ODE fresh on every call."
)


def _eval_tau_dense(
    interp: dfx.DenseInterpolation,
    tau_unit: u.AbstractUnit,
    s_max: u.AbstractQuantity,
    s: u.AbstractQuantity,
    /,
) -> u.AbstractQuantity:
    r"""Evaluate a precomputed $\tau(s)$ interpolation at ``s``.

    Valid over $[-m,\, s_{\max} + m]$, the range `_solve_tau_dense` actually
    integrated (see `_S_MAX_MARGIN`); beyond it this raises, rather than
    letting `diffrax.DenseInterpolation` extrapolate off the end of its
    coefficients.
    """
    s_unit = s_max.unit
    s_val = jnp.asarray(s.ustrip(s_unit))
    s_max_val = jnp.asarray(s_max.ustrip(s_unit))

    margin = _S_MAX_MARGIN * jnp.abs(s_max_val)
    # A NaN `s` is False for both out-of-domain tests, so negate in-domain.
    out_of_domain = ~((s_val >= -margin) & (s_val <= s_max_val + margin))

    # No clip: every point of the solved range is real data, so the only
    # thing to do with a genuine overshoot is refuse it.
    s_val = eqx.error_if(s_val, out_of_domain, _MSG_S_OUT_OF_DOMAIN)
    return u.Q(interp.evaluate(s_val), tau_unit)


def _arclength_between(
    curve: Callable[[Any], Any],
    tau_unit: u.AbstractUnit,
    s_unit: u.AbstractUnit,
    tau_0: u.AbstractQuantity,
    tau: u.AbstractQuantity,
    diffeqsolver: DiffEqSolver,
    /,
) -> Any:
    r"""$S = \int_{\tau_0}^{\tau} \|\boldsymbol{\gamma}'(u)\|\,du$, as a bare float.

    The one extra quadrature `_tau_of_s`'s JVP rule needs -- see its docstring
    for the derivation. Returned unit-stripped (in ``s_unit``) because it is
    only ever fed straight into a cotangent, never displayed on its own.
    Rescaled to $\sigma \in [0, 1]$ for the reason `_solve_tau` is, which
    applies here to ``tau_0``.
    """
    # `jnp.asarray` narrows only here, as in ``nearest.py``: `ustrip` is typed as
    # a broad union that `-` is not defined across, though every runtime member
    # supports it.
    tau_0_val = jnp.asarray(tau_0.ustrip(tau_unit))
    tau_val = jnp.asarray(tau.ustrip(tau_unit))
    dtau = tau_val - tau_0_val

    args = (curve, tau_0_val, dtau, tau_unit, s_unit)
    sol = diffeqsolver(_TERM_QUAD, 0.0, 1.0, None, 0.0, args)
    return sol.ys[-1]


def _tangent_value(t: Any, /) -> Any:
    """Unwrap a tangent to a bare value; a symbolic zero has no leaves."""
    return next(iter(jax.tree_util.tree_leaves(t)), 0.0)


def _tau_primal(
    curve: Callable[[Any], Any],
    tau_0: u.AbstractQuantity,
    s: u.AbstractQuantity,
    interp: dfx.DenseInterpolation | None,
    s_max: u.AbstractQuantity | None,
    /,
    *,
    tau_unit: u.AbstractUnit,
    diffeqsolver: DiffEqSolver,
) -> u.AbstractQuantity:
    r"""Compute the primal $\tau(s)$, shared by `_tau_of_s` and its JVP rule."""
    if interp is not None:
        return _eval_tau_dense(interp, tau_unit, cast("u.AbstractQuantity", s_max), s)
    return _solve_tau(curve, tau_unit, tau_0, diffeqsolver, s)


@eqx.filter_custom_jvp
def _tau_of_s(
    curve: Callable[[Any], Any],
    tau_0: u.AbstractQuantity,
    s: u.AbstractQuantity,
    interp: dfx.DenseInterpolation | None,
    s_max: u.AbstractQuantity | None,
    /,
    *,
    tau_unit: u.AbstractUnit,
    diffeqsolver: DiffEqSolver,
) -> u.AbstractQuantity:
    r"""Solve for $\tau(s)$, with a hand-supplied JVP in place of autodiff.

    ``curve`` here is already one-argument (`ArcLength` passes its own
    ``curve``; `LagrangianArcLength` passes ``AtTime(curve, t0)``).

    **Why a custom JVP rather than `custom_vjp`.** Two independent reasons.
    Autodiff through a precomputed ``interp`` cannot reach ``curve``'s leaves
    at all -- the coefficients are numbers by then, computed from the *old*
    curve -- so a perturbation via `equinox.tree_at` silently loses the
    $\partial\tau/\partial\theta$ term (issue #713 has the measured
    numbers). And `BishopBuilder` differentiates the curve it wraps in
    forward mode (``bishop.py``'s `_tangent_at` uses `unxt.experimental.jacfwd`),
    which a `custom_vjp` cannot support -- ``bishop.py:68`` documents the same
    trap for `RecursiveCheckpointAdjoint`. A custom JVP serves forward mode
    directly and reverse mode by transposition, so one rule covers both.

    From $s = S(\tau; \theta) = \int_{\tau_0}^{\tau}
    \|\boldsymbol{\gamma}'(u; \theta)\|\,du$, differentiating at fixed $s$
    gives $\|\boldsymbol{\gamma}'(\tau)\|\,d\tau - ds + dS = 0$, i.e.

    $$ d\tau = \frac{ds - dS}{\|\boldsymbol{\gamma}'(\tau;\theta)\|}, $$

    with $dS$ the differential of $S$ at fixed $\tau$ in the directions
    $(d\theta, d\tau_0)$. Taking it as one total differential rather than
    three partials means a *single* `equinox.filter_jvp` of one quadrature
    (`_arclength_between`) covers $\theta$, $\tau_0$ (Leibniz on the lower
    limit) and $s$ alike. ``curve`` and ``tau_0`` are always the *current*
    call's values, never anything captured at interpolation-build time.

    This makes the *gradient* correct at the point ``curve``/``tau_0`` hold,
    which is what #713's acceptance criteria check. It does not make a stale
    interpolation's forward *value* correct far from that point; nothing can,
    short of rebuilding it (see `ArcLength.s_max`).
    """
    return _tau_primal(
        curve, tau_0, s, interp, s_max, tau_unit=tau_unit, diffeqsolver=diffeqsolver
    )


@_tau_of_s.def_jvp
def _tau_of_s_jvp(
    primals: tuple[Any, u.AbstractQuantity, u.AbstractQuantity, Any, Any],
    tangents: tuple[Any, Any, Any, Any, Any],
    *,
    tau_unit: u.AbstractUnit,
    diffeqsolver: DiffEqSolver,
) -> tuple[u.AbstractQuantity, u.AbstractQuantity]:
    r"""Apply the hand-supplied JVP -- see `_tau_of_s` for the formula and why.

    ``interp`` and ``s_max`` are positional (not keyword) arguments to
    `_tau_of_s` specifically so their tangents land here rather than
    tripping `equinox.filter_custom_jvp`'s "keyword arguments are always
    nondifferentiable" check: when `ArcLength` is built *inside* the
    function being differentiated, ``interp`` genuinely depends on the
    differentiated leaves (it was computed from them). Their tangents
    (``dinterp``, ``ds_max``) are simply unused below -- the whole point of
    this custom rule is that $\partial\tau/\partial\theta$ never routes
    through ``interp`` at all, only through the analytic formula.
    """
    curve, tau_0, s, interp, s_max = primals
    dcurve, dtau_0, ds, _dinterp, _ds_max = tangents

    tau = _tau_primal(
        curve, tau_0, s, interp, s_max, tau_unit=tau_unit, diffeqsolver=diffeqsolver
    )
    s_unit = s.unit
    speed_at_tau = _speed(curve, tau_unit, s_unit, tau)

    # dS is the only reason this rule integrates anything, so skip the solve
    # when neither the curve nor tau_0 is differentiated -- the common case,
    # covering every derivative in `s` alone (chart Jacobian, induced metric,
    # `nearest_tau`'s root-find). Symbolically-zero tangents have no leaves,
    # a *structure* question resolved at trace time, so this stays jit-safe;
    # a merely numerically-zero tangent still has a leaf and takes the full
    # path.
    if jax.tree_util.tree_leaves((dcurve, dtau_0)):

        def arclength(c: Any, t0: u.AbstractQuantity) -> Any:
            return _arclength_between(c, tau_unit, s_unit, t0, tau, diffeqsolver)

        _, dS = eqx.filter_jvp(arclength, (curve, tau_0), (dcurve, dtau_0))
        dS_val = _tangent_value(dS)
    else:
        dS_val = 0.0

    # Both terms are bare values in `s_unit`, so the subtraction is safe
    # without a conversion: `dS_val` comes from `_arclength_between`, which
    # integrates in `s_unit` by construction, and `ds_val` inherits `s`'s own
    # unit because a JVP tangent shares its primal's pytree structure -- and
    # a `Quantity`'s unit lives in the treedef, not the leaf.
    ds_val = _tangent_value(ds)
    dtau_val = (ds_val - dS_val) / speed_at_tau
    return tau, u.Q(dtau_val, tau_unit)


_MSG_MISSING_TIME = (
    "this `ArcLength` wraps a two-argument curve `gamma(tau, t)`, so it must be "
    "called as `arc(s, t)`. Bind the time first with `AtTime(arc, t)` to get a "
    "one-argument curve."
)

_MSG_UNINSPECTABLE_CURVE = (
    "could not inspect the signature of the given curve to detect whether it "
    "is one-argument `tau -> ...` or two-argument `(tau, t) -> ...`. The "
    "curve must be a plain Python callable with an inspectable signature "
    "`(tau)` or `(tau, t)`, not a builtin or C-implemented callable."
)

_MSG_REQUIRED_KEYWORD_ONLY = (
    "the given curve's second parameter `{name}` is keyword-only and has no "
    "default, so the curve can be called neither as `curve(tau)` (it is "
    "missing) nor as `curve(tau, t)` (it cannot be reached positionally). "
    "Give it a default if it is a tuning knob, make it positional if it is "
    "the time, or bind it with `functools.partial(curve, {name}=...)`."
)

_MSG_LAGRANGIAN_REQUIRES_TWO_ARGUMENT = (
    "LagrangianArcLength wraps a two-argument curve `gamma(tau, t)`, but the "
    "given curve does not accept two required positional arguments. A "
    "one-argument curve has no distinct time slices for `t0` to fix -- wrap "
    "it in `ArcLength` instead."
)

_MSG_S_MAX_TWO_ARGUMENT = (
    "s_max is not supported for a two-argument (time-dependent) curve. "
    "ArcLength's Eulerian reading re-measures arc length on whichever slice "
    "`t` it is evaluated at, so there is no single tau(s) map to precompute "
    "-- the map genuinely differs per t. Bind `t` first with "
    "`AtTime(curve, t)`, which freezes the slice and makes the wrapped "
    "curve one-argument, or leave `s_max=None`."
)


def _is_two_argument(curve: Callable[..., Any], /) -> bool:
    """Report whether ``curve`` takes ``(tau, t)`` rather than just ``(tau)``.

    A wrapper that *knows* its own arity is asked first, via a
    ``_two_argument`` attribute. `ArcLength` defaults ``t`` in its
    ``__call__`` so that a wrapped one-argument curve can still be called
    ``arc(s)`` -- convenient, but it makes the signature describe the
    wrapper rather than what it wraps, and a builder consulting that
    signature concludes "one-argument" for a wrapper that genuinely needs a
    time. `AtTime` needs no such attribute: it binds the time, so being
    one-argument is the truth about it (see #748).

    Otherwise the signature is read. The question is then what the *call*
    accepts, so the second parameter must be both **positional** and
    **required**. Weakening either half misreads ordinary signatures:

    ============================  ====================  ==================
    signature                     ``curve(tau)`` works  reading
    ============================  ====================  ==================
    ``(tau, t)``                  no                    two-argument
    ``(tau, smoothing=0.1)``      yes                   one-argument
    ``partial(curve, t=...)``     yes                   one-argument
    ``(tau, *args)``              yes                   one-argument
    ``(tau, **kw)``               yes                   one-argument
    ``(tau, *, resolution)``      no                    neither -- raises
    ============================  ====================  ==================

    The defaulted cases are why "required" is checked rather than counting
    parameters: `inspect.signature` keeps a `functools.partial`-bound
    parameter, with a default. The variadic cases are why "positional" is
    checked rather than "required" alone -- ``*args`` and ``**kw`` have no
    default, yet ``curve(tau)`` binds them empty and works. Reading either as
    time-dependent sends the curve down the two-argument path, where it fails
    deep inside the ODE solve, nowhere near the real cause.

    A **required keyword-only** second parameter is neither reading: it
    cannot be reached positionally, so ``curve(tau, t)`` fails, and it has no
    default, so ``curve(tau)`` fails too. That is a broken curve rather than
    an ambiguous one, so it raises here with its own message instead of being
    forced into a reading that cannot work.
    """
    declared = getattr(curve, "_two_argument", None)
    if isinstance(declared, bool):
        return declared

    try:
        params = list(inspect.signature(curve).parameters.values())
    except (TypeError, ValueError) as e:
        raise TypeError(_MSG_UNINSPECTABLE_CURVE) from e

    if len(params) < 2:
        return False
    second = params[1]

    if second.kind in (
        inspect.Parameter.VAR_POSITIONAL,  # *args: curve(tau) still binds
        inspect.Parameter.VAR_KEYWORD,  # **kw: likewise
    ):
        return False
    if second.default is not inspect.Parameter.empty:
        return False
    if second.kind is inspect.Parameter.KEYWORD_ONLY:
        raise TypeError(_MSG_REQUIRED_KEYWORD_ONLY.format(name=second.name))
    return True


@final
class ArcLength(eqx.Module):
    r"""Reparameterise a curve by arc length.

    Wraps a curve $\boldsymbol{\gamma}(\tau)$ and returns a curve
    $s \mapsto \boldsymbol{\gamma}(\tau(s))$ with unit speed, by solving

    $$ \frac{d\tau}{ds} = \frac{1}{\|\boldsymbol{\gamma}'(\tau)\|},
    \qquad \tau(0) = \tau_0 $$

    from $s = 0$ to the requested $s$, rather than integrating speed and
    inverting. The result is differentiable through the solve in both AD
    modes (see `diffeqsolver`).

    If the wrapped ``curve`` also takes an evaluation time, i.e. it is
    ``(tau, t) -> Quantity`` rather than ``tau -> Quantity``, `ArcLength`
    detects this at construction and **stays two-argument**:
    ``ArcLength(curve)(s, t)`` measures arc length on the slice at ``t`` --
    the *Eulerian* reading. This is different from binding ``t`` first with
    `AtTime` and wrapping the result: ``ArcLength(AtTime(curve, t))`` is a
    one-argument curve frozen to that single slice. See `AtTime` for the
    distinction and why the order matters.

    Parameters
    ----------
    curve : Callable
        A function ``tau -> Quantity[float, (3,)]`` or, for a time-dependent
        curve, ``(tau, t) -> Quantity[float, (3,)]``, in the parameter unit
        ``tau_unit``.  Make it an `equinox.Module` for differentiable curve
        parameters; a bare function's captures are trace-time constants.
    tau_unit : AbstractUnit or str
        Unit of the *wrapped* curve's parameter $\tau$ -- a time for a
        time-parametrised curve, a length for one already parametrised by
        arc length. Not the unit of $s$, which is instead read off the
        length `Quantity` passed to `__call__`.
    tau_0 : Quantity, optional
        Reference parameter where $s = 0$. Defaults to ``Q(0.0, tau_unit)``.
    diffeqsolver : DiffEqSolver, optional
        `diffraxtra.DiffEqSolver` configuring the ODE solve: solver,
        step-size controller, adjoint and step budget in one object. A
        **static** field (it holds no arrays), so changing it recompiles
        rather than retracing silently; see `BishopBuilder`'s *Changing one
        knob* for how to derive from the default with `dataclasses.replace`.
    s_max : Quantity, optional
        When given, the reparametrisation ODE is solved **once**, at
        construction, as a dense interpolation of $\tau(s)$ over $s \in [0,
        s_{\max}]$; `__call__` then reads that instead of re-solving. A pure
        performance knob: values agree to solver tolerance and gradients
        agree exactly, because `__call__` always reparametrises through the
        hand-supplied JVP in `_tau_of_s` rather than autodiff through the
        solve or the interpolation. `None` (the default) solves fresh every
        call. A query outside $[0, s_{\max}]$ raises rather than
        extrapolating (up to `_S_MAX_MARGIN`'s slack at either end).

        Only valid for a **one-argument** ``curve``: the Eulerian reading
        re-measures arc length per slice, so there is no single map to
        precompute (use `LagrangianArcLength`, whose reference slice is
        fixed). Passing it with a two-argument ``curve`` raises here.

        `TubularChart.tau_bounds` bounds the same curve parameter at a
        different layer -- what `nearest_tau` scans -- and the two do not
        update each other. Set `s_max` to at least `tau_bounds[1]` in the
        same unit, or in-bounds chart queries will land outside the
        interpolation and raise.

    See Also
    --------
    coordinaxs.curveframes.LagrangianArcLength :
        Always measures arc length on a fixed reference slice, rather than
        the slice being evaluated.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def helix(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")

    >>> arc = cxfc.ArcLength(helix, "s")
    >>> arc(u.Q(0.0, "km"))
    Q([1., 0., 0.], 'km')

    """

    curve: Callable[[Any], Any]
    """The wrapped curve."""

    tau_unit: u.AbstractUnit = eqx.field(static=True)
    """Unit of the wrapped curve's parameter tau.

    A time for a time-parametrised curve, a length for one already
    parametrised by arc length.
    """

    tau_0: u.AbstractQuantity
    """Reference parameter value where s = 0 (a leaf).

    A `None` passed to `__init__` resolves to ``Q(0.0, tau_unit)``.
    """

    diffeqsolver: DiffEqSolver = eqx.field(static=True)
    """Solver, step-size controller, adjoint and step budget for the ODE.

    Static, as in `BishopBuilder`: the solver's tolerances and step budget are
    configuration, not data. Left dynamic they become pytree leaves, so a
    `jax.tree.map` over the module silently rescales `rtol`/`atol` and can turn
    `max_steps` into a tracer.
    """

    s_max: u.AbstractQuantity | None
    """If given, precompute tau(s) once over s in [0, s_max] (a leaf).

    `None` (the default) keeps the per-call solve. Only valid for a
    one-argument ``curve``; see the class docstring.
    """

    #: Dimension of the parameter this wrapper *exposes*. Reparametrising by
    #: arc length makes it a length, whatever the wrapped curve was
    #: parametrised by -- so a builder over this can check its `tau_unit`
    #: against it at construction rather than failing later (#718).
    _param_dimension: ClassVar[str] = "length"

    _two_argument: bool = eqx.field(static=True)
    """Whether ``curve`` takes ``(tau, t)`` rather than just ``tau``.

    Detected once from ``curve``'s signature in ``__init__`` (see
    `_is_two_argument`).
    """

    _interp: dfx.DenseInterpolation | None
    """Dense interpolation of tau(s) over [0, s_max], or `None`.

    Built once in ``__init__`` when ``s_max`` is given (see
    `_solve_tau_dense`); left `None` otherwise, in which case `__call__`
    solves fresh every time. Gradients bypass this field entirely -- see
    `_tau_of_s`.
    """

    # An explicit `__init__` rather than `__post_init__`: `_interp` is a
    # dynamic field built from the other arguments, and `equinox` both warns
    # about `field(init=False)` on one of those and applies field converters
    # only *after* `__post_init__`, where `_solve_tau_dense` already needs the
    # converted `tau_unit`.
    def __init__(
        self,
        curve: Callable[[Any], Any],
        tau_unit: u.AbstractUnit | str,
        tau_0: u.AbstractQuantity | None = None,
        diffeqsolver: DiffEqSolver | None = None,
        s_max: u.AbstractQuantity | None = None,
    ) -> None:
        """See the class docstring for the parameters."""
        self.curve = curve
        self.tau_unit = u.unit(tau_unit)  # ty: ignore[invalid-assignment]
        self.tau_0 = tau_0 if tau_0 is not None else u.Q(0.0, self.tau_unit)
        self.diffeqsolver = diffeqsolver if diffeqsolver is not None else _DIFFEQSOLVER
        self.s_max = s_max
        self._two_argument = _is_two_argument(curve)

        if s_max is None:
            self._interp = None
        elif self._two_argument:
            raise ValueError(_MSG_S_MAX_TWO_ARGUMENT)
        else:
            self._interp = _solve_tau_dense(
                curve, self.tau_unit, self.tau_0, self.diffeqsolver, s_max
            )

    def __call__(self, s: u.AbstractQuantity, t: Any = None, /) -> Any:
        r"""Evaluate the reparameterised curve at arc length ``s``.

        If the wrapped curve is time-dependent, ``t`` selects the slice on
        which arc length is measured (the Eulerian reading); it is ignored
        (and may be omitted) for a one-argument curve.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "km")

        >>> arc = cxfc.ArcLength(circle, "s")
        >>> jnp.round(arc(u.Q(0.0, "km")).value, 6)
        Array([1., 0., 0.], dtype=float64)

        """
        tau_unit = self.tau_unit
        tau_0 = self.tau_0

        # Bind `t` into a one-argument curve for a time-dependent wrapped
        # curve; otherwise use it as-is. See `_two_argument`'s docstring for
        # why this is a static-field branch rather than a per-call one.
        if self._two_argument and t is None:
            # Without this the omission surfaces as an `AttributeError` on
            # `None` from inside the ODE, nowhere near the call that caused it.
            raise TypeError(_MSG_MISSING_TIME)
        curve = self.curve if not self._two_argument else AtTime(self.curve, t)

        tau = _tau_of_s(
            curve,
            tau_0,
            s,
            self._interp,
            self.s_max,
            tau_unit=tau_unit,
            diffeqsolver=self.diffeqsolver,
        )
        return curve(tau)


@final
class LagrangianArcLength(eqx.Module):
    r"""Reparameterise a time-dependent curve by arc length on a fixed slice.

    Wraps a two-argument curve $\boldsymbol{\gamma}(\tau, t)$ and returns a
    curve $(s, t) \mapsto \boldsymbol{\gamma}(\tau(s), t)$ where $\tau(s)$
    solves

    $$ \frac{d\tau}{ds} = \frac{1}{\|\partial_\tau\boldsymbol{\gamma}(\tau, t_0)\|},
    \qquad \tau(0) = \tau_0, $$

    i.e. arc length is always measured on the **fixed** reference slice
    $t_0$, never on the slice ``t`` supplied at call time. The solved $\tau$
    is then plugged into $\boldsymbol{\gamma}(\cdot, t)$ at that *supplied*
    time. A label therefore names a fixed material point -- however the
    curve has since moved -- rather than a position on the current curve.

    Parameters
    ----------
    curve : Callable
        A function ``(tau, t) -> Quantity[float, (3,)]``, in the parameter
        unit ``tau_unit``. Make it an `equinox.Module` for differentiable
        curve parameters; a bare function's captures are trace-time
        constants.
    t0 : Quantity
        The fixed reference slice on which arc length is measured.
    tau_unit : AbstractUnit or str
        Unit of the wrapped curve's parameter $\tau$ -- a time for a
        time-parametrised curve, a length for one already parametrised by
        arc length. Not the unit of $s$, which is instead read off the
        length `Quantity` passed to `__call__`.
    tau_0 : Quantity, optional
        Reference parameter where $s = 0$. Defaults to ``Q(0.0, tau_unit)``.
    diffeqsolver : DiffEqSolver, optional
        `diffraxtra.DiffEqSolver` configuring the ODE solve; see `ArcLength`.
    s_max : Quantity, optional
        When given, precompute tau(s) once as a dense interpolation over $s
        \in [0, s_{\max}]$, exactly as `ArcLength.s_max` does; see there for
        the full behaviour. Unlike `ArcLength`, this is always valid here --
        the reference slice ``t0`` is fixed, so the map genuinely does not
        depend on the ``t`` supplied at call time. `None` (the default) keeps
        the per-call solve.

    See Also
    --------
    coordinaxs.curveframes.ArcLength :
        The *Eulerian* reading of the same two-argument curve: it re-measures
        arc length on whichever slice is evaluated, rather than a fixed one.
        The two readings agree exactly under rigid motion and diverge once
        the curve stretches or bends.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def stretch(tau: u.Q, t: u.Q) -> u.Q:
    ...     x = tau.ustrip("s") * (1.0 + 0.5 * t.ustrip("s"))
    ...     z = jnp.zeros_like(x)
    ...     return u.Q(jnp.stack([x, z, z]), "km")

    >>> lag = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"), "s")
    >>> lag(u.Q(1.0, "km"), u.Q(1.0, "s"))
    Q([1.5, 0. , 0. ], 'km')

    """

    #: Same as `ArcLength`: the exposed parameter is an arc length, though
    #: measured on the fixed reference slice `t0` rather than the current one.
    _param_dimension: ClassVar[str] = "length"

    curve: Callable[[Any, Any], Any]
    """The wrapped, two-argument curve."""

    t0: u.AbstractQuantity
    """The fixed reference slice on which arc length is measured."""

    tau_unit: u.AbstractUnit = eqx.field(static=True)
    """Unit of the wrapped curve's parameter tau.

    A time for a time-parametrised curve, a length for one already
    parametrised by arc length.
    """

    tau_0: u.AbstractQuantity
    """Reference parameter value where s = 0 (a leaf).

    A `None` passed to `__init__` resolves to ``Q(0.0, tau_unit)``.
    """

    diffeqsolver: DiffEqSolver = eqx.field(static=True)
    """Solver, step-size controller, adjoint and step budget for the ODE.

    Static, as in `BishopBuilder`: the solver's tolerances and step budget are
    configuration, not data. Left dynamic they become pytree leaves, so a
    `jax.tree.map` over the module silently rescales `rtol`/`atol` and can turn
    `max_steps` into a tracer.
    """

    s_max: u.AbstractQuantity | None
    """If given, precompute tau(s) once over s in [0, s_max] (a leaf).

    `None` (the default) keeps the per-call solve. Always valid here, unlike
    `ArcLength.s_max`: the reference slice ``t0`` is fixed, so the map does
    not depend on the ``t`` supplied at call time.
    """

    _interp: dfx.DenseInterpolation | None
    """Dense interpolation of tau(s) over [0, s_max], or `None`.

    Built once in ``__init__`` when ``s_max`` is given (see
    `_solve_tau_dense`), against the fixed reference slice ``t0``; left
    `None` otherwise. See `ArcLength._interp` for why gradients do not rely
    on this field's own connectedness to `curve`/`t0`.
    """

    # Explicit `__init__` for the reasons `ArcLength.__init__` gives.
    def __init__(
        self,
        curve: Callable[[Any, Any], Any],
        t0: u.AbstractQuantity,
        tau_unit: u.AbstractUnit | str,
        tau_0: u.AbstractQuantity | None = None,
        diffeqsolver: DiffEqSolver | None = None,
        s_max: u.AbstractQuantity | None = None,
    ) -> None:
        """See the class docstring for the parameters."""
        if not _is_two_argument(curve):
            raise TypeError(_MSG_LAGRANGIAN_REQUIRES_TWO_ARGUMENT)
        self.curve = curve
        self.t0 = t0
        self.tau_unit = u.unit(tau_unit)  # ty: ignore[invalid-assignment]
        self.tau_0 = tau_0 if tau_0 is not None else u.Q(0.0, self.tau_unit)
        self.diffeqsolver = diffeqsolver if diffeqsolver is not None else _DIFFEQSOLVER
        self.s_max = s_max

        if s_max is None:
            self._interp = None
        else:
            curve_t0 = AtTime(curve, t0)
            self._interp = _solve_tau_dense(
                curve_t0, self.tau_unit, self.tau_0, self.diffeqsolver, s_max
            )

    def __call__(self, s: u.AbstractQuantity, t: Any, /) -> Any:
        r"""Evaluate the reparameterised curve at label ``s`` and time ``t``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def stretch(tau: u.Q, t: u.Q) -> u.Q:
        ...     x = tau.ustrip("s") * (1.0 + 0.5 * t.ustrip("s"))
        ...     z = jnp.zeros_like(x)
        ...     return u.Q(jnp.stack([x, z, z]), "km")

        >>> lag = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"), "s")
        >>> lag(u.Q(1.0, "km"), u.Q(0.0, "s"))
        Q([1., 0., 0.], 'km')

        """
        tau_unit = self.tau_unit
        tau_0 = self.tau_0

        # Speed is always measured on the fixed slice t0 -- never on the
        # supplied t -- which is what makes this reading Lagrangian.
        curve_t0 = AtTime(self.curve, self.t0)
        tau = _tau_of_s(
            curve_t0,
            tau_0,
            s,
            self._interp,
            self.s_max,
            tau_unit=tau_unit,
            diffeqsolver=self.diffeqsolver,
        )

        # But the resulting tau is evaluated on the *supplied* slice, so the
        # label rides with the material point as the curve moves.
        return self.curve(tau, t)
