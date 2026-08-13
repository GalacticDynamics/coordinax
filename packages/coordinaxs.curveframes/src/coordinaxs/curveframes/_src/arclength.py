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

__all__ = ("ArcLength",)

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

    tau_unit: u.AbstractUnit = eqx.field(  # ty: ignore[invalid-assignment]
        default=u.unit("s"), static=True, converter=u.unit
    )
    """The unit of the wrapped curve's parameter tau."""

    tau_0: u.AbstractQuantity | None = None
    """Reference parameter value where s = 0 (a leaf).

    `None` is resolved to ``Q(0.0, tau_unit)`` by ``__post_init__``.
    """

    # See `BishopBuilder.diffeqsolver` for why this is static and why the
    # default is a factory rather than a plain default.
    diffeqsolver: DiffEqSolver = eqx.field(
        default_factory=lambda: _DIFFEQSOLVER, static=True
    )
    """Solver, step-size controller, adjoint and step budget for the ODE."""

    _two_argument: bool = eqx.field(static=True, init=False, default=False)
    """Whether ``curve`` takes ``(tau, t)`` rather than just ``tau``.

    Detected once from ``curve``'s signature in ``__post_init__`` (see
    `_is_two_argument`) so `__call__` never re-inspects ``curve``.
    """

    def __post_init__(self) -> None:
        """Resolve a `None` ``tau_0`` to zero in ``tau_unit`` (a pytree leaf)."""
        if self.tau_0 is None:
            self.tau_0 = u.Q(0.0, self.tau_unit)
        self._two_argument = _is_two_argument(self.curve)

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
        tau_0 = cast("u.AbstractQuantity", self.tau_0)
        s_unit = s.unit
        speed_unit = s_unit / tau_unit

        # Bind `t` into a one-argument curve for a time-dependent wrapped
        # curve; otherwise use it as-is. `self._two_argument` is a static
        # field, so this is a Python-level branch resolved once per traced
        # curve shape, not a per-call runtime branch under jit.
        curve = self.curve if not self._two_argument else AtTime(self.curve, t)

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

        sol = self.diffeqsolver(dfx.ODETerm(ode_rhs), 0.0, 1.0, None, tau_0_val)
        tau_val = sol.ys[-1]
        return curve(u.Q(tau_val, tau_unit))
