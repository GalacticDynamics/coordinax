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
`.location` are the only two call sites), `ArcLength(curve)` is itself a curve
and can be wrapped in any of the existing frame builders with no further
change: e.g. ``BishopBuilder(ArcLength(curve))``.

"""

__all__ = ("ArcLength", "LagrangianArcLength")

import inspect

from collections.abc import Callable
from typing import Any, cast, final

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
_DIFFEQSOLVER = DiffEqSolver(
    solver=dfx.Tsit5(),
    stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10),
    adjoint=dfx.DirectAdjoint(),
    max_steps=16384,
)


def _speed(
    curve: Callable[[Any], Any],
    tau_unit: u.AbstractUnit,
    speed_unit: u.AbstractUnit,
    tau_q: u.AbstractQuantity,
    /,
) -> Any:
    r"""Local speed $\|\boldsymbol{\gamma}'(\tau)\|$ of ``curve`` at ``tau_q``."""
    dcurve = u.experimental.jacfwd(curve, units=(tau_unit,))
    return jnp.linalg.norm(dcurve(tau_q).ustrip(speed_unit))


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
    speed_unit = s_unit / tau_unit
    s_val = s.ustrip(s_unit)
    tau_0_val = tau_0.ustrip(tau_unit)

    def ode_rhs(sigma: Any, tau_flat: Any, args: Any) -> Any:
        """Right-hand side in the rescaled parameter ``sigma``."""
        del sigma, args
        tau_q = u.Q(tau_flat, tau_unit)
        return s_val / _speed(curve, tau_unit, speed_unit, tau_q)

    sol = diffeqsolver(dfx.ODETerm(ode_rhs), 0.0, 1.0, None, tau_0_val)
    return u.Q(sol.ys[-1], tau_unit)


def _solve_tau_dense(
    curve: Callable[[Any], Any],
    tau_unit: u.AbstractUnit,
    tau_0: u.AbstractQuantity,
    diffeqsolver: DiffEqSolver,
    s_max: u.AbstractQuantity,
    /,
) -> dfx.DenseInterpolation:
    r"""Solve the reparametrisation ODE once, densely, over $[0, s_{\max}]$.

    Companion to `_solve_tau`: the same ODE and the same right-hand side, but
    solved a single time over the whole domain with `diffrax`'s dense output
    (``SaveAt(dense=True)``) rather than once per requested ``s``. This is
    what makes the *forward* evaluation cheap under repeated calls at a fixed
    curve -- see `ArcLength.s_max`.

    Unlike `_solve_tau`, this does *not* rescale to $\sigma \in [0, 1]$. That
    rescaling exists to keep $d(\text{result})/ds$ well-defined when ``s`` --
    the value being differentiated -- is also the integration bound: at
    ``s = 0`` the solver would otherwise take zero steps and silently return
    a zero derivative. Here the integration bound is the fixed ``s_max``, not
    the ``s`` a caller later evaluates at, and the returned interpolation is
    only ever consumed through `_tau_of_s`'s custom VJP, never
    autodifferentiated directly -- see that function for why.
    """
    s_unit = s_max.unit
    speed_unit = s_unit / tau_unit
    tau_0_val = tau_0.ustrip(tau_unit)
    s_max_val = s_max.ustrip(s_unit)

    def ode_rhs(sigma: Any, tau_flat: Any, args: Any) -> Any:
        del sigma, args
        tau_q = u.Q(tau_flat, tau_unit)
        return 1.0 / _speed(curve, tau_unit, speed_unit, tau_q)

    sol = diffeqsolver(
        dfx.ODETerm(ode_rhs),
        0.0,
        s_max_val,
        None,
        tau_0_val,
        saveat=dfx.SaveAt(dense=True),
    )
    return cast("dfx.DenseInterpolation", sol.interpolation)


#: Relative margin (a fraction of `s_max`) that `_eval_tau_dense` clamps
#: into silently rather than raising on. Sized to the *structural* slack
#: `nearest_tau` needs, not floating-point noise: its bracketed root-find
#: (`nearest.py`) evaluates the curve at `tau0 +/- spacing`, one full scan
#: seed spacing outside `tau_bounds`, for *every* query whose nearest seed
#: lands at a domain edge -- not a rare event, but the normal case for a
#: point near either end of the curve. Default `n_seed=64` over one period
#: gives spacing ~= 1.6% of the range; 5% comfortably covers that with room
#: to spare down to `n_seed ~= 20`. A caller running `nearest_tau` with a
#: much smaller `n_seed` against a tightly-fit `s_max` may need a wider
#: margin than this fixed fraction gives -- widen `s_max` itself, which is
#: also what `TubularChart.tau_bounds` should already recommend (see
#: `ArcLength.s_max`).
_S_MAX_MARGIN = 0.05

_MSG_S_OUT_OF_DOMAIN = (
    "s lies outside the precomputed domain [0, s_max] by more than the "
    "margin _eval_tau_dense tolerates. The dense interpolation only has "
    "values there -- diffrax would otherwise return NaN silently for a "
    "query this far outside it. Increase s_max, or leave it `None` to fall "
    "back to solving the ODE fresh on every call."
)


def _eval_tau_dense(
    interp: dfx.DenseInterpolation,
    tau_unit: u.AbstractUnit,
    s_max: u.AbstractQuantity,
    s: u.AbstractQuantity,
    /,
) -> u.AbstractQuantity:
    r"""Evaluate a precomputed $\tau(s)$ interpolation at ``s``.

    ``s`` should lie in $[0, s_{\max}]$, the domain `_solve_tau_dense` built
    the interpolation over. An exact ``[0, s_max]`` gate is too strict: a
    legitimate caller can land measurably outside it, not just by
    floating-point noise -- e.g. `nearest_tau`'s own bracketed root-find,
    which evaluates the curve one full scan-seed spacing beyond
    `tau_bounds` for any query near a domain edge (see `TubularChart`,
    measured there at ``s = -4.73e-9`` for a *converged* result, and
    considerably more for the root-find's own intermediate probes). Within
    `_S_MAX_MARGIN` of the domain, this clamps and evaluates at the
    boundary rather than raising -- a converged, in-tolerance answer should
    not fail just because the search that found it briefly stepped outside
    while getting there. Only a query genuinely beyond that margin raises,
    rather than letting `diffrax.DenseInterpolation` clamp internally and
    return `NaN` silently.
    """
    s_unit = s_max.unit
    s_val = jnp.asarray(s.ustrip(s_unit))
    s_max_val = jnp.asarray(s_max.ustrip(s_unit))

    slack = _S_MAX_MARGIN * jnp.maximum(1.0, jnp.abs(s_max_val))
    out_of_domain = (s_val < -slack) | (s_val > s_max_val + slack)

    s_clipped = jnp.clip(s_val, 0.0, s_max_val)
    s_clipped = eqx.error_if(s_clipped, out_of_domain, _MSG_S_OUT_OF_DOMAIN)
    return u.Q(interp.evaluate(s_clipped), tau_unit)


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

    The one extra quadrature `_tau_of_s`'s backward rule needs -- see its
    docstring for the implicit-function-theorem derivation this feeds into.
    Returned as a plain (unit-stripped, in ``s_unit``) value rather than a
    `unxt.Quantity`: it is only ever scaled and fed straight into a cotangent,
    never displayed or unit-checked on its own.

    Rescaled to $\sigma \in [0, 1]$ exactly as `_solve_tau` is, so this stays
    differentiable in ``tau_0`` -- one of the two things `_tau_of_s`'s
    backward rule differentiates it against -- even when ``tau_0 == tau``
    (the ``s = 0`` case): integrating directly over ``[tau_0, tau]`` would put
    a differentiated quantity in the integration *bound*, where the solver
    takes zero steps and the derivative silently comes back as zero.
    """
    speed_unit = s_unit / tau_unit
    # `jnp.asarray` narrows only here, as in `nearest.py`: `ustrip` is typed as
    # a broad union that `-` is not defined across, though every runtime member
    # supports it.
    tau_0_val = jnp.asarray(tau_0.ustrip(tau_unit))
    tau_val = jnp.asarray(tau.ustrip(tau_unit))
    dtau = tau_val - tau_0_val

    def ode_rhs(sigma: Any, s_flat: Any, args: Any) -> Any:
        del s_flat, args
        tau_q = u.Q(tau_0_val + sigma * dtau, tau_unit)
        return dtau * _speed(curve, tau_unit, speed_unit, tau_q)

    sol = diffeqsolver(dfx.ODETerm(ode_rhs), 0.0, 1.0, None, 0.0)
    return sol.ys[-1]


def _tangent_value(t: Any, /) -> Any:
    r"""Unwrap a `filter_custom_jvp` tangent to a bare float, `0.0` if symbolic zero.

    A tangent that is symbolically zero -- an argument not being
    differentiated -- arrives as `None` for a bare array, or (since
    `unxt.Quantity` is a pytree with one array child) as a `Quantity` whose
    *wrapped value* is `None`. Either way it has no leaves, so
    `jax.tree_util.tree_leaves` reduces both to the empty list; a genuine
    tangent always has exactly one.
    """
    leaves = jax.tree_util.tree_leaves(t)
    return leaves[0] if leaves else 0.0


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

    **Why a custom JVP, and not the more obviously-named `custom_vjp`.** Two
    separate problems, one fix each:

    1. When ``interp`` is given, the primal below is just `_eval_tau_dense`
       reading off a `diffrax.DenseInterpolation` built once, at
       `ArcLength.__init__` (see `ArcLength.s_max`), from whatever ``curve``
       looked like *then*. If a caller later perturbs ``curve``'s own leaves
       -- e.g. via `equinox.tree_at`, replacing a curve parameter without
       rebuilding the `ArcLength` -- ordinary autodiff through
       `interp.evaluate` would see no path back to that leaf at all: the
       interpolation's coefficients are just numbers by that point, computed
       from the *old* curve. The gradient would silently come back missing
       the $\partial\tau/\partial\theta$ contribution entirely (this is what
       #713 measured: a cached gradient of 1.9156525704 against a correct
       -1.7574794224, and `LagrangianArcLength.t0`'s gradient returning
       exactly 0.0). No amount of care in `__init__` fixes this -- a
       precomputed leaf is structurally stale with respect to perturbations
       of the inputs it was built from. Fixed by not asking autodiff to
       differentiate through the interpolation at all, and supplying the
       true derivative directly (below).

    2. `BishopBuilder` differentiates the curve it wraps in *forward* mode
       too (`_tangent_at` uses `unxt.experimental.jacfwd`) -- that is the
       entire reason it defaults to `diffrax.DirectAdjoint` rather than
       `diffrax`'s own default, which is a `custom_vjp` and so cannot be
       `jvp`-ed (see `BishopBuilder`'s *Choosing an adjoint*). A
       `jax.custom_vjp`-wrapped reparametrisation would break exactly the
       same way, the moment it sits inside a curve handed to `BishopBuilder`
       -- measured: ``TypeError: can't apply forward-mode autodiff (jvp) to a
       custom_vjp function``. `jax.custom_jvp` does not have this problem,
       and JAX derives a correct, cheap reverse-mode rule from it
       automatically (by transposing the linear tangent formula below), so
       one hand-written rule serves both AD modes.

    From $s = S(\tau; \theta) = \int_{\tau_0}^{\tau}
    \|\boldsymbol{\gamma}'(u; \theta)\|\,du$, the implicit function theorem
    gives, at fixed $s$:

    $$ \frac{\partial \tau}{\partial \theta}
       = -\frac{(\partial S/\partial\theta)|_\tau}{\|\boldsymbol{\gamma}'(\tau;
         \theta)\|}, \qquad
       \frac{\partial \tau}{\partial \tau_0}
       = \frac{\|\boldsymbol{\gamma}'(\tau_0;\theta)\|}{\|\boldsymbol{\gamma}'(\tau;
         \theta)\|}, \qquad
       \frac{\partial \tau}{\partial s} = \frac{1}{\|\boldsymbol{\gamma}'(\tau;
         \theta)\|}. $$

    Rather than build each of these separately, the JVP rule below computes
    the total differential in one pass: given tangents $d\theta$, $d\tau_0$,
    $ds$, the total derivative of $S(\tau;\theta,\tau_0) = s$ at fixed $\tau$
    is $dS = (\partial S/\partial\theta)\,d\theta +
    (\partial S/\partial\tau_0)\,d\tau_0 = \|\boldsymbol{\gamma}'(\tau)\|\,
    d\tau - ds$ rearranges to

    $$ d\tau = \frac{ds - dS}{\|\boldsymbol{\gamma}'(\tau;\theta)\|}, $$

    where $dS$ is obtained with a *single* `equinox.filter_jvp` of one
    quadrature (`_arclength_between`) in the directions $(d\theta,
    d\tau_0)$ -- cheap against reverse-mode through nested adaptive ODE
    solves, which is what this replaces (the $\tau_0$ formula above is what
    that JVP reduces to automatically when only $d\tau_0$ is nonzero --
    Leibniz's rule on a variable lower limit -- not a separate special
    case). ``curve`` and ``tau_0`` here are always the *current* call's
    values, never anything captured at interpolation-build time.

    This is correct *at* the point ``curve``/``tau_0`` were built with --
    exactly where a gradient (a local, linear quantity) is asked to be
    correct, and exactly what #713's acceptance criteria check. It does not
    make a *stale* interpolation's forward *value* correct far from that
    point; nothing can, short of rebuilding it (see `ArcLength.s_max`).
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
    speed_unit = s_unit / tau_unit
    speed_at_tau = _speed(curve, tau_unit, speed_unit, tau)

    # dS/dtheta is the *only* reason this rule integrates anything. When
    # neither the curve nor tau_0 is being differentiated, both tangents are
    # symbolically zero, dS is identically zero, and the quadrature below is
    # a solve whose result is discarded.
    #
    # That is not a corner case: it is every derivative taken with respect to
    # `s` alone, which includes the chart's Jacobian, the induced metric that
    # pulls back through it, and `nearest_tau`'s root-find. Those all carry a
    # fixed curve.
    #
    # Emptiness of `tree_leaves` is the symbolic-zero test `_tangent_value`
    # already relies on, and it is a *structure* question, resolved at trace
    # time -- so this branch is chosen during tracing and stays jit-safe. A
    # tangent that merely happens to be numerically zero still has a leaf,
    # and correctly takes the full path.
    differentiating_curve = bool(jax.tree_util.tree_leaves((dcurve, dtau_0)))

    if differentiating_curve:

        def arclength(c: Any, t0: u.AbstractQuantity) -> Any:
            return _arclength_between(c, tau_unit, s_unit, t0, tau, diffeqsolver)

        _, dS = eqx.filter_jvp(arclength, (curve, tau_0), (dcurve, dtau_0))
        dS_val = _tangent_value(dS)
    else:
        dS_val = 0.0

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
    """Report whether ``curve`` takes two positional arguments, ``(tau, t)``.

    The second parameter must be **required**. Counting parameters instead
    misreads two ordinary idioms as time-dependent curves: a one-argument
    curve carrying a tuning knob, ``def curve(tau, smoothing=0.1)``, and a
    curve whose time a caller has already frozen with
    ``functools.partial(curve, t=...)`` -- `inspect.signature` keeps a
    keyword-bound parameter, with a default. Both then receive ``t=None``
    and fail deep inside the ODE solve, nowhere near the real cause.
    """
    try:
        params = list(inspect.signature(curve).parameters.values())
    except (TypeError, ValueError) as e:
        raise TypeError(_MSG_UNINSPECTABLE_CURVE) from e
    return len(params) >= 2 and params[1].default is inspect.Parameter.empty


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
        s_{\max}]$ (`diffrax`'s ``SaveAt(dense=True)``); `__call__` then
        evaluates that interpolation instead of re-solving. This is a pure
        performance knob: within $[0, s_{\max}]$, evaluating with and without
        `s_max` agrees to solver tolerance, and gradients agree too --
        `__call__`'s reparametrisation always goes through a hand-supplied
        VJP (`_tau_of_s`) rather than autodiff through the solve or the
        interpolation, specifically so that swapping the fast path in does
        not touch gradient correctness. See `_tau_of_s` for why that is
        necessary, not just convenient. The default `None` keeps today's
        behaviour exactly -- solve fresh on every call, no precompute.

        Only valid for a **one-argument** ``curve``; passing it alongside a
        two-argument (time-dependent) ``curve`` raises at construction,
        because the Eulerian reading re-measures arc length per slice and so
        has no single map to precompute (see `LagrangianArcLength` for the
        two-argument case that *can* be precomputed). A query outside $[0,
        s_{\max}]$ raises rather than extrapolating (with a small tolerance
        for floating-point slack at the endpoints -- see `_eval_tau_dense`).

        **Interaction with `TubularChart.tau_bounds`.** Both describe the
        same valid range of the curve parameter, at two different layers:
        `TubularChart.tau_bounds` bounds what `nearest_tau` will scan and
        return when inverting a Cartesian point into this chart, while
        `s_max` bounds what the precomputed interpolation covers. They do
        not update each other, and can disagree. Set `s_max` to at least
        `tau_bounds[1]` (in the same unit) so no in-bounds chart query lands
        outside the interpolation's domain; a smaller `s_max` will raise on
        exactly the queries a chart considers valid.

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

    Precomputing the reparametrisation over a bounded range of ``s`` trades
    memory for speed on repeated calls, with no change in behaviour:

    >>> fast_arc = cxfc.ArcLength(helix, "s", s_max=u.Q(10.0, "km"))
    >>> fast_arc(u.Q(0.0, "km"))
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

    diffeqsolver: DiffEqSolver
    """Solver, step-size controller, adjoint and step budget for the ODE."""

    s_max: u.AbstractQuantity | None
    """If given, precompute tau(s) once over s in [0, s_max] (a leaf).

    `None` (the default) keeps the per-call solve. Only valid for a
    one-argument ``curve``; see the class docstring.
    """

    _two_argument: bool = eqx.field(static=True)
    """Whether ``curve`` takes ``(tau, t)`` rather than just ``tau``.

    Detected once from ``curve``'s signature in ``__init__`` (see
    `_is_two_argument`).
    """

    _interp: dfx.DenseInterpolation | None
    """Dense interpolation of tau(s) over [0, s_max], or `None`.

    Built once in ``__init__`` when ``s_max`` is given (see
    `_solve_tau_dense`); left `None` otherwise, in which case `__call__`
    solves fresh every time. Gradients never flow through this field
    directly -- see `_tau_of_s` for why an `equinox.field(init=False)` field
    (the natural way to spell "derived from other fields") is not enough on
    its own, and why a plain field set from a custom `__init__` (this
    module's own workaround, matching `AtTime`'s) is not enough either, and
    what actually fixes it.
    """

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
        the full behaviour, including gradient correctness and the
        interaction with `TubularChart.tau_bounds`. Unlike `ArcLength`, this
        is always valid here -- the reference slice ``t0`` is fixed, so the
        map genuinely does not depend on the ``t`` supplied at call time.
        `None` (the default) keeps the per-call solve.

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

    diffeqsolver: DiffEqSolver
    """Solver, step-size controller, adjoint and step budget for the ODE."""

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
