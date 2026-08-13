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

    # Pre-compute the curve's derivative as a callable, once, rather than
    # nesting AD inside the ODE right-hand side.
    dcurve = u.experimental.jacfwd(curve, units=(tau_unit,))

    def speed(tau_q: u.AbstractQuantity, /) -> Any:
        return jnp.linalg.norm(dcurve(tau_q).ustrip(speed_unit))

    s_val = s.ustrip(s_unit)
    tau_0_val = tau_0.ustrip(tau_unit)

    def ode_rhs(sigma: Any, tau_flat: Any, args: Any) -> Any:
        """Right-hand side in the rescaled parameter ``sigma``."""
        del sigma, args
        tau_q = u.Q(tau_flat, tau_unit)
        return s_val / speed(tau_q)

    sol = diffeqsolver(dfx.ODETerm(ode_rhs), 0.0, 1.0, None, tau_0_val)
    return u.Q(sol.ys[-1], tau_unit)


def _is_two_argument(curve: Callable[..., Any], /) -> bool:
    """Report whether ``curve`` takes two positional arguments, ``(tau, t)``.

    Inspected once, at `ArcLength` construction, and cached in a static
    field -- so `__call__` branches on a Python `bool` (resolved at trace
    time, one compiled variant per curve shape) rather than re-inspecting
    ``curve`` on every call.

    The second parameter must be **required**. Counting parameters instead
    misreads two ordinary idioms as time-dependent curves: a one-argument
    curve carrying a tuning knob, ``def curve(tau, smoothing=0.1)``, and a
    curve whose time a caller has already frozen with
    ``functools.partial(curve, t=...)`` -- `inspect.signature` keeps a
    keyword-bound parameter, with a default. Both then receive ``t=None``
    and fail deep inside the ODE solve, nowhere near the real cause.
    """
    params = list(inspect.signature(curve).parameters.values())
    return len(params) >= 2 and params[1].default is inspect.Parameter.empty


_MSG_S_MAX_TWO_ARGUMENT = (
    "s_max is not supported for a two-argument (time-dependent) curve. "
    "ArcLength's Eulerian reading re-measures arc length on whichever slice "
    "`t` it is evaluated at, so there is no single tau(s) map to precompute "
    "-- the map genuinely differs per t. Bind `t` first with "
    "`AtTime(curve, t)`, which freezes the slice and makes the wrapped "
    "curve one-argument, or leave `s_max=None`."
)

_MSG_S_OUT_OF_DOMAIN = (
    "s lies outside the precomputed domain [0, s_max]. The dense "
    "interpolation is only valid there -- diffrax would otherwise return "
    "NaN silently for a query outside it. Increase s_max, or leave it "
    "`None` to fall back to solving the ODE fresh on every call."
)


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
    (``SaveAt(dense=True)``) rather than once per requested ``s``.

    Unlike `_solve_tau`, this does *not* rescale to $\sigma \in [0, 1]$. That
    rescaling exists to keep $d(\text{result})/ds$ well-defined when ``s`` --
    the value being differentiated -- is also the integration bound: at
    ``s = 0`` the solver would otherwise take zero steps and silently return
    a zero derivative. Here the integration bound is the fixed ``s_max``, not
    the ``s`` a caller later evaluates at; the returned interpolation is a
    smooth function of $s$ everywhere in its domain, ``s_max`` included, so
    the same pathology does not arise.
    """
    s_unit = s_max.unit
    speed_unit = s_unit / tau_unit

    # Pre-compute the curve's derivative as a callable, once, rather than
    # nesting AD inside the ODE right-hand side.
    dcurve = u.experimental.jacfwd(curve, units=(tau_unit,))

    def speed(tau_q: u.AbstractQuantity, /) -> Any:
        return jnp.linalg.norm(dcurve(tau_q).ustrip(speed_unit))

    tau_0_val = tau_0.ustrip(tau_unit)
    s_max_val = s_max.ustrip(s_unit)

    def ode_rhs(sigma: Any, tau_flat: Any, args: Any) -> Any:
        del sigma, args
        tau_q = u.Q(tau_flat, tau_unit)
        return 1.0 / speed(tau_q)

    sol = diffeqsolver(
        dfx.ODETerm(ode_rhs),
        0.0,
        s_max_val,
        None,
        tau_0_val,
        saveat=dfx.SaveAt(dense=True),
    )
    return cast("dfx.DenseInterpolation", sol.interpolation)


def _eval_tau_dense(
    interp: dfx.DenseInterpolation,
    tau_unit: u.AbstractUnit,
    s_max: u.AbstractQuantity,
    s: u.AbstractQuantity,
    /,
) -> u.AbstractQuantity:
    r"""Evaluate a precomputed $\tau(s)$ interpolation at ``s``.

    ``s`` must lie in $[0, s_{\max}]$, the domain `_solve_tau_dense` built
    the interpolation over. Outside it, `diffrax.DenseInterpolation` clamps
    the query and returns `NaN` rather than raising; `equinox.error_if`
    turns that into a clear, jit-compatible error instead of letting a wrong
    (`NaN`) answer through silently.
    """
    s_unit = s_max.unit
    s_val = s.ustrip(s_unit)
    # `jnp.asarray` narrows only for `ty`: `ustrip` is typed as a broad union and
    # `<`/`>` are not defined across all of it. Same narrowing as `nearest.py`.
    s_max_val = jnp.asarray(s_max.ustrip(s_unit))
    s_arr = jnp.asarray(s_val)
    s_val = eqx.error_if(
        s_val, (s_arr < 0.0) | (s_arr > s_max_val), _MSG_S_OUT_OF_DOMAIN
    )
    return u.Q(interp.evaluate(s_val), tau_unit)


@final
class ArcLength(eqx.Module):
    r"""Reparameterise a curve by arc length.

    Wraps a curve $\boldsymbol{\gamma}(\tau)$ and returns a curve
    $s \mapsto \boldsymbol{\gamma}(\tau(s))$ with unit speed, by solving

    $$ \frac{d\tau}{ds} = \frac{1}{\|\boldsymbol{\gamma}'(\tau)\|},
    \qquad \tau(0) = \tau_0 $$

    from $s = 0$ to the requested $s$, rather than integrating speed and
    inverting.  The result is differentiable through the solve in both AD
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
    tau_unit : AbstractUnit or str, optional
        Unit of the *wrapped* curve's parameter $\tau$ -- not of $s$, which is
        instead read off the length `Quantity` passed to `__call__`. Defaults
        to ``"s"``.
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
        evaluates that interpolation instead of re-solving. This only
        changes performance, not behaviour: within the domain the two
        agree to solver tolerance. Calling with ``s`` outside $[0,
        s_{\max}]$ raises rather than extrapolating silently. The default
        `None` keeps today's behaviour exactly -- solve fresh on every
        call, no precompute. Only valid for a **one-argument** ``curve``;
        passing it alongside a two-argument (time-dependent) ``curve``
        raises at construction, because the Eulerian reading re-measures
        arc length per slice and so has no single map to precompute (see
        `LagrangianArcLength` for the two-argument case that *can* be
        precomputed). Pass a `Quantity` to make it a differentiable leaf, or
        a `StaticQuantity` to keep it out of the pytree.

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

    >>> arc = cxfc.ArcLength(helix)
    >>> arc(u.Q(0.0, "km"))
    Q([1., 0., 0.], 'km')

    """

    curve: Callable[[Any], Any]
    """The wrapped curve."""

    tau_unit: u.AbstractUnit = eqx.field(static=True)
    """The unit of the wrapped curve's parameter tau."""

    tau_0: u.AbstractQuantity
    """Reference parameter value where s = 0 (a leaf).

    A `None` passed to `__init__` resolves to ``Q(0.0, tau_unit)``.
    """

    diffeqsolver: DiffEqSolver = eqx.field(static=True)
    """Solver, step-size controller, adjoint and step budget for the ODE."""

    s_max: u.AbstractQuantity | None
    """If given, precompute tau(s) once over s in [0, s_max] (a leaf).

    `None` (the default) keeps the per-call solve. Only valid for a
    one-argument ``curve``; see the class docstring.
    """

    _two_argument: bool = eqx.field(static=True)
    """Whether ``curve`` takes ``(tau, t)`` rather than just ``tau``.

    Detected once from ``curve``'s signature in ``__init__`` (see
    `_is_two_argument`) so `__call__` never re-inspects ``curve``.
    """

    _interp: dfx.DenseInterpolation | None
    """Dense interpolation of tau(s) over [0, s_max], or `None`.

    Built once in ``__init__`` when ``s_max`` is given (see
    `_solve_tau_dense`); left `None` otherwise, in which case `__call__`
    solves fresh every time.

    Not an `equinox.field(init=False)` field, even though `__init__` always
    sets it and never accepts it: `equinox.Module` warns that an
    `init=False` field holding array leaves silently breaks gradients
    w.r.t. the fields it was derived from (its value would come back as an
    independent, disconnected leaf under ``jax.grad``, rather than as a
    function of ``curve`` and ``s_max``). Computing it in a custom
    `__init__` instead -- the same trick `AtTime` uses for
    ``_curve_dynamic`` -- sidesteps the warning *and* the bug: assignment
    happens while `curve` is still a normal Python value being traced, so
    gradients flow through it like any other derived quantity.
    """

    def __init__(
        self,
        curve: Callable[[Any], Any],
        tau_unit: u.AbstractUnit | str = "s",
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

        Solves $d\tau/ds = 1/\|\boldsymbol{\gamma}'(\tau)\|$ from $s = 0$ to
        ``s``, over the rescaled parameter $\sigma \in [0, 1]$ with
        $s(\sigma) = \sigma \cdot s_{\mathrm{val}}$. As in
        `BishopBuilder._solve_U1`, the rescaling is what keeps the solve
        differentiable in ``s`` at ``s = 0``: integrating over $[0, s]$
        directly would put ``s`` in the integration bound, where the solver
        loop takes zero steps and the derivative silently comes back as $0$.

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

        >>> arc = cxfc.ArcLength(circle)
        >>> jnp.round(arc(u.Q(0.0, "km")).value, 6)
        Array([1., 0., 0.], dtype=float64)

        """
        tau_unit = self.tau_unit
        tau_0 = self.tau_0

        # Bind `t` into a one-argument curve for a time-dependent wrapped
        # curve; otherwise use it as-is. `self._two_argument` is a static
        # field, so this is a Python-level branch resolved once per traced
        # curve shape, not a per-call runtime branch under jit.
        curve = self.curve if not self._two_argument else AtTime(self.curve, t)

        if self._interp is not None:
            tau = _eval_tau_dense(self._interp, tau_unit, self.s_max, s)
        else:
            tau = _solve_tau(curve, tau_unit, tau_0, self.diffeqsolver, s)
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

    This is the *Lagrangian* reading, and it differs from `ArcLength` over
    the same two-argument curve, which stays *Eulerian*: it re-measures arc
    length on whichever slice ``t`` it is evaluated at, so its label tracks
    the current curve rather than a material point. The two readings agree
    exactly when the curve moves rigidly (no slice's arc length differs from
    any other's); they diverge once the curve stretches or bends.

    Parameters
    ----------
    curve : Callable
        A function ``(tau, t) -> Quantity[float, (3,)]``, in the parameter
        unit ``tau_unit``. Make it an `equinox.Module` for differentiable
        curve parameters; a bare function's captures are trace-time
        constants.
    t0 : Quantity
        The fixed reference slice on which arc length is measured.
    tau_unit : AbstractUnit or str, optional
        Unit of the wrapped curve's parameter $\tau$ -- not of $s$, which is
        instead read off the length `Quantity` passed to `__call__`. Defaults
        to ``"s"``.
    tau_0 : Quantity, optional
        Reference parameter where $s = 0$. Defaults to ``Q(0.0, tau_unit)``.
    diffeqsolver : DiffEqSolver, optional
        `diffraxtra.DiffEqSolver` configuring the ODE solve; see `ArcLength`.
    s_max : Quantity, optional
        When given, precompute tau(s) once as a dense interpolation over $s
        \in [0, s_{\max}]$, exactly as `ArcLength.s_max` does; see there for
        the full behaviour. Unlike `ArcLength`, this is always valid here --
        the reference slice ``t0`` is fixed, so the map genuinely does not
        depend on the ``t`` supplied at call time. `None` (the default)
        keeps the per-call solve.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def stretch(tau: u.Q, t: u.Q) -> u.Q:
    ...     x = tau.ustrip("s") * (1.0 + 0.5 * t.ustrip("s"))
    ...     z = jnp.zeros_like(x)
    ...     return u.Q(jnp.stack([x, z, z]), "km")

    >>> lag = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"))
    >>> lag(u.Q(1.0, "km"), u.Q(1.0, "s"))
    Q([1.5, 0. , 0. ], 'km')

    """

    curve: Callable[[Any, Any], Any]
    """The wrapped, two-argument curve."""

    t0: u.AbstractQuantity
    """The fixed reference slice on which arc length is measured."""

    tau_unit: u.AbstractUnit = eqx.field(static=True)
    """The unit of the wrapped curve's parameter tau."""

    tau_0: u.AbstractQuantity
    """Reference parameter value where s = 0 (a leaf).

    A `None` passed to `__init__` resolves to ``Q(0.0, tau_unit)``.
    """

    diffeqsolver: DiffEqSolver = eqx.field(static=True)
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
    `None` otherwise, in which case `__call__` solves fresh every time.

    See `ArcLength._interp` for why this is a plain field, set only from a
    custom `__init__`, rather than an `equinox.field(init=False)` one.
    """

    def __init__(
        self,
        curve: Callable[[Any, Any], Any],
        t0: u.AbstractQuantity,
        tau_unit: u.AbstractUnit | str = "s",
        tau_0: u.AbstractQuantity | None = None,
        diffeqsolver: DiffEqSolver | None = None,
        s_max: u.AbstractQuantity | None = None,
    ) -> None:
        """See the class docstring for the parameters."""
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

        Solves for $\tau(s)$ against the curve's speed on the fixed slice
        ``t0``, then evaluates the wrapped curve at ``(tau(s), t)``. See
        `ArcLength.__call__` for why the ODE is rescaled to $\sigma \in
        [0, 1]$.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def stretch(tau: u.Q, t: u.Q) -> u.Q:
        ...     x = tau.ustrip("s") * (1.0 + 0.5 * t.ustrip("s"))
        ...     z = jnp.zeros_like(x)
        ...     return u.Q(jnp.stack([x, z, z]), "km")

        >>> lag = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"))
        >>> lag(u.Q(1.0, "km"), u.Q(0.0, "s"))
        Q([1., 0., 0.], 'km')

        """
        tau_unit = self.tau_unit
        tau_0 = self.tau_0

        # Speed is always measured on the fixed slice t0 -- never on the
        # supplied t -- which is what makes this reading Lagrangian.
        if self._interp is not None:
            tau = _eval_tau_dense(self._interp, tau_unit, self.s_max, s)
        else:
            curve_t0 = AtTime(self.curve, self.t0)
            tau = _solve_tau(curve_t0, tau_unit, tau_0, self.diffeqsolver, s)

        # But the resulting tau is evaluated on the *supplied* slice, so the
        # label rides with the material point as the curve moves.
        return self.curve(tau, t)
