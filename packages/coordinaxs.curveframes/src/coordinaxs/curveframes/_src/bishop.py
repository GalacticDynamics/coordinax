r"""Bishop (rotation-minimising) curve-frame data types.

This module provides the concrete implementations of the Bishop
(parallel-transport, rotation-minimising) curve-frame apparatus:

* `BishopBuilder` — an `equinox.Module` mapping $\tau$ to the rigid-body
  transform ``Translate(-\gamma) | Rotate([T; U1; U2])``.
* `BishopFrame` — a curve-attached reference frame whose axes are $(\mathbf{T},
  \mathbf{U}_1, \mathbf{U}_2)$ obtained by parallel transport along the curve.

Unlike the Frenet--Serret frame, the Bishop frame is **well-defined even when
the curvature vanishes** ($\kappa = 0$), because it does not depend on
$\boldsymbol{\gamma}''$.  The normal-plane vectors are obtained by solving the
parallel-transport ODE:

$$ \frac{d\mathbf{U}_i}{d\tau}
  = -\bigl(\mathbf{U}_i \cdot \mathbf{T}'\bigr)\,\mathbf{T}, \qquad i \in \{1,
  2\},
$$

starting from an initial orthonormal pair at a reference parameter $\tau_0$.
The ODE is integrated numerically with `diffrax`, by default using a
`diffrax.DirectAdjoint` so the solve is differentiable in **both** modes:
reverse-mode for gradients w.r.t. curve parameters, and forward-mode for the
tangent/jet propagation that `coordinax.transforms.act` and
`coordinax.transforms.act_jet` need.  Solver, adjoint, step-size controller and
step budget travel together in `BishopBuilder`'s single `diffraxtra.DiffEqSolver`
field, so a caller can trade accuracy for speed; see that class's *Choosing an
adjoint* section for the one trade-off that changes what the frame can *do*
rather than how fast it does it.

Both classes are ``@final`` (no further subclassing).

Key design choices
------------------
* **Lazy evaluation** — the ODE is solved only when a concrete $\tau$ is
  requested, i.e. when the `TimeDep` family is evaluated.
* **Auto initial normal** — when no ``initial_normal`` is supplied, one is
  chosen automatically via Gram--Schmidt against the tangent at $\tau_0$.

See Also
--------
coordinaxs.curveframes._src.frenetserret : Frenet--Serret frame.
coordinaxs.curveframes._src.base : Abstract base classes.

"""

__all__ = ("BishopBuilder", "BishopFrame")

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, final

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffraxtra import DiffEqSolver

import coordinax.transforms as cxfm
import unxt as u

from .arclength import _is_two_argument
from .base import (
    AbstractCurveFrameBuilder,
    AbstractParallelTransportFrame,
    FrameT,
    unit_or_none,
)
from .frenetserret import _normalize

#: Default integrator for the parallel-transport ODE.  `DirectAdjoint` is the
#: only adjoint that is differentiable in *both* modes: the `diffrax` default
#: `RecursiveCheckpointAdjoint` (like `jax.experimental.ode.odeint`) is a
#: `custom_vjp` and so cannot be `jvp`-ed, which is exactly what the tangent /
#: jet machinery does; `BacksolveAdjoint` is unusable here in *either* mode,
#: because the reparametrised right-hand side closes over ``dtau`` /
#: ``tau_0_val`` and its backwards solve cannot carry them; and `ForwardMode`
#: gives up reverse mode.  At these tolerances orthonormality of the resulting
#: R holds to ~9e-12 over ``|tau| <= 60`` on a helix, against ~9e-9 for the
#: `odeint` solve this replaces.  ``max_steps`` is raised from `diffrax`'s 4096,
#: which is tight: a unit-radius helix at these tolerances takes ~20 steps per
#: unit of ``|tau - tau_0|``, so the default caps transport at ``|dtau| ~ 200``.
#: Measured to cost nothing in either compile or run time; a longer transport
#: raises rather than silently truncating.
#:
#: Every field is restated even where it matches `diffrax`, because this object
#: -- not `diffrax`'s signature -- is what a caller starts from: the documented
#: way to change one knob is `dataclasses.replace` on this default, which keeps
#: the other three.  See `BishopBuilder`'s ``diffeqsolver`` field.
_DIFFEQSOLVER = DiffEqSolver(
    solver=dfx.Tsit5(),
    stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10),
    adjoint=dfx.DirectAdjoint(),
    max_steps=16384,
)

_MSG_BATCH_TWO_ARGUMENT = (
    "`rotation_matrices` needs a one-argument curve: for a two-argument one "
    "each tau selects a different time slice, so the parameters do not share "
    "an ODE solve and there is nothing to batch. Use `rotation_matrix` per tau, "
    "or `jax.vmap` over it."
)

_MSG_BATCH_RANK = (
    "`rotation_matrices` takes a 1-D batch of parameters; got shape {shape}. "
    "`diffrax.SaveAt` saves on a 1-D grid, so a higher-rank batch has no "
    "single ordering to solve along. Flatten it, or use `rotation_matrix` for "
    "a single parameter."
)

_MSG_STRADDLES_TAU_0 = (
    "`rotation_matrices` needs every tau on one side of tau_0: the transport "
    "runs outward from tau_0 in a single monotonic solve, and a set that "
    "straddles it would need two. Split the parameters and call twice."
)

_MSG_PARALLEL_NORMAL = (
    "`initial_normal` is parallel to the tangent at tau_0; it has no component "
    "in the normal plane. Pass a vector that is not along the tangent, or leave "
    "it `None` to have one chosen automatically."
)


def _float(x: Any, /) -> Array:
    """Convert to an array, promoting integers to float but preserving f32.

    ``dtype=float`` would name the *default* float, silently widening an f32
    input to f64 under ``jax_enable_x64`` and discarding a deliberate choice of
    single precision. `jnp.result_type` promotes only what needs promoting.
    """
    arr = jnp.asarray(x)
    return arr.astype(jnp.result_type(arr, float))


def _orthonormalize(v: Any, T0_val: Any) -> Any:
    r"""Gram--Schmidt ``v`` against the unit tangent, then normalise.

    Raises (via `equinox.error_if`, so it also fires under ``jit``) when ``v``
    is parallel to ``T0_val`` and the rejection vanishes.  The test is on the
    rejection *relative* to ``|v|``, so it measures the angle between the two
    vectors rather than the length of ``v``: the result is exactly homogeneous
    of degree zero in ``v``, and a small-but-valid ``v`` must not be rejected.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinaxs.curveframes._src.bishop import _orthonormalize

    >>> T0 = jnp.array([1.0, 0.0, 0.0])
    >>> _orthonormalize(jnp.array([2.0, 3.0, 0.0]), T0)
    Array([0., 1., 0.], dtype=float64)

    Scaling ``v`` down does not change the answer:

    >>> _orthonormalize(jnp.array([0.0, 0.0, 1e-12]), T0)
    Array([0., 0., 1.], dtype=float64)

    """
    w = v - jnp.dot(v, T0_val) * T0_val
    norm = jnp.linalg.norm(w)
    # `~(norm > tol)`, not `norm <= tol`: a NaN compares False against both, so
    # the `<=` form admits a NaN `v` and returns a NaN triad with nothing raised.
    # Same reason `TubularChart`'s reach guard is written `~(f > 0)`. Negating
    # `>` keeps the non-strict sense, so an all-zero `v` (threshold 0) still
    # raises.
    w = eqx.error_if(w, ~(norm > 1e-12 * jnp.linalg.norm(v)), _MSG_PARALLEL_NORMAL)
    return w / norm


def _auto_initial_normal(T0_val: Any) -> Any:
    r"""Choose an initial unit normal to the unit tangent $\mathbf{T}_0$.

    Gram--Schmidt against the standard basis vector least aligned with
    $\mathbf{T}_0$, which keeps the rejection well-conditioned even when
    $\mathbf{T}_0$ nearly coincides with a coordinate axis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinaxs.curveframes._src.bishop import _auto_initial_normal

    When the tangent is along x, the initial normal avoids x:

    >>> T0 = jnp.array([1.0, 0.0, 0.0])
    >>> U1 = _auto_initial_normal(T0)
    >>> float(jnp.dot(U1, T0))
    0.0
    >>> float(jnp.linalg.norm(U1))
    1.0

    """
    # Pick the standard basis axis least aligned with T0.
    # This maximises the rejection magnitude and avoids numerical
    # cancellation in the Gram--Schmidt step.
    abs_T0 = jnp.abs(T0_val)
    k = jnp.argmin(abs_T0)
    e_k = jnp.zeros(3).at[k].set(1.0)
    return _orthonormalize(e_k, T0_val)


@final
class BishopBuilder(AbstractCurveFrameBuilder):
    r"""Bishop (rotation-minimising) frame family along a curve.

    The Bishop frame attaches an orthonormal triad $(\mathbf{T}, \mathbf{U}_1,
    \mathbf{U}_2)$ to each point of a smooth space curve $\gamma(\tau)$ via
    parallel transport:

    - $\mathbf{T}$ (tangent): unit tangent vector $\gamma'/\|\gamma'\|$
    - $\mathbf{U}_1$ (normal 1): parallel-transported first normal
    - $\mathbf{U}_2$ (normal 2): $\mathbf{T} \times \mathbf{U}_1$

    Unlike the Frenet-Serret frame, the Bishop frame is well-defined even when
    the curvature vanishes ($\kappa = 0$).

    Calling the builder at $\tau$ returns ``Translate(-gamma) | Rotate(R)`` with
    $R = [\mathbf{T};\,\mathbf{U}_1;\,\mathbf{U}_2]$.

    Parameters
    ----------
    curve : Callable
        A function ``tau -> Quantity[float, (3,)]``.  Make it an
        `equinox.Module` for differentiable curve parameters.
    tau_unit : AbstractUnit or str, optional
        Unit of the curve parameter.  `None` (the default) reads it off the
        parameter the builder is called with.  There is no neutral unit to
        default to -- a curve parameter may be a time, an arc length, or an
        affine parameter -- so rather than pick one, take the one the caller
        already stated by passing a `Quantity`.  Declare it for a curve that
        reads its argument's ``.value`` rather than converting, or for a raw
        (unitless) parameter or ``station``, neither of which carries a unit
        to read.
    station : optional
        A fixed station along the curve; see `AbstractCurveFrameBuilder`.
    tau_0 : Quantity, optional
        Reference parameter where the initial frame is defined.  Defaults to
        ``Q(0.0, tau_unit)``.
    initial_normal : array-like, optional
        Dimensionless 3-vector for $\mathbf{U}_{1,0}$.  When `None`,
        auto-chosen via Gram--Schmidt against the tangent at ``tau_0``.
    diffeqsolver : DiffEqSolver, optional
        `diffraxtra.DiffEqSolver` configuring the parallel-transport solve:
        solver, step-size controller, adjoint and step budget in one object.
        A **static** field (it holds no arrays), so changing it recompiles
        rather than retracing silently.  See *Changing one knob*.

    Notes
    -----
    **Changing one knob.** ``DiffEqSolver``'s *own* field defaults are read off
    `diffrax.diffeqsolve`'s signature, and all of them are wrong for this
    solve -- its default adjoint in particular is `RecursiveCheckpointAdjoint`,
    which silently costs forward mode (see below).  So derive from *this*
    builder's ``diffeqsolver`` with `dataclasses.replace` (or
    `equinox.tree_at`, which reaches inside the ``DiffEqSolver`` but not the
    builder, whose field is static and so not a leaf) rather than constructing
    a ``DiffEqSolver`` from scratch; see the Examples.

    **Choosing an adjoint.** The default is `diffrax.DirectAdjoint`, *not*
    `diffrax`'s own default, because the tangent/jet machinery differentiates
    the solve in forward mode:

    ============================ ============ ============ =======
    Adjoint                      forward (AD) reverse (AD) speed
    ============================ ============ ============ =======
    `DirectAdjoint` (default)    yes          yes          slowest
    `RecursiveCheckpointAdjoint` **no**       yes          fastest
    `ForwardMode`                yes          **no**       fast
    `BacksolveAdjoint`           **no**       **no**       fast
    ============================ ============ ============ =======

    `BacksolveAdjoint` is unusable here in *either* mode: the reparametrised
    right-hand side closes over ``dtau`` and ``tau_0_val``, which its
    backwards solve cannot carry, so both directions raise JAX's
    ``CustomVJPException`` ("...with respect to a closed-over value").  That
    is a property of the reparametrisation, not of the curve -- it fails the
    same way for a curve written as a bare function with no array leaves at
    all, so there is no way to write the curve that recovers it.

    For `RecursiveCheckpointAdjoint`, "forward: no" means
    `coordinax.transforms.act` on tangent data and
    `coordinax.transforms.act_jet` raise ``TypeError: can't apply forward-mode
    autodiff (jvp) to a custom_vjp function`` -- accurate, but it never names
    the adjoint you chose.  So pick it only when you want `grad` and nothing
    else; it is otherwise a silent loss of capability.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    Define a helix:

    >>> def helix(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

    >>> bt = cxfc.BishopBuilder(helix, "s")
    >>> bt.location(u.Q(0.0, "s"))
    Q([1., 0., 0.], 'm')

    Loosen the tolerances and keep everything else, `DirectAdjoint` included:

    >>> import dataclasses
    >>> import diffrax as dfx
    >>> fast = dataclasses.replace(
    ...     bt,
    ...     diffeqsolver=dataclasses.replace(
    ...         bt.diffeqsolver,
    ...         stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
    ...     ),
    ... )
    >>> type(fast.diffeqsolver.adjoint).__name__, fast.diffeqsolver.max_steps
    ('DirectAdjoint', 16384)

    The unit may be left off, in which case it is read off the parameter --
    which is a `Quantity`, so it states its own unit already:

    >>> cxfc.BishopBuilder(helix).tangent(u.Q(0.0, "s"))
    Q([-0.        ,  0.70710678,  0.70710678], '')

    Declaring it is still accepted and gives the same answer:

    >>> cxfc.BishopBuilder(helix, "s").tangent(u.Q(0.0, "s"))
    Q([-0.        ,  0.70710678,  0.70710678], '')

    The Bishop frame works on a straight line (where Frenet-Serret is singular):

    >>> def line(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([t, jnp.zeros_like(t),
    ...                           jnp.zeros_like(t)]), "m")

    >>> U1 = cxfc.BishopBuilder(line, "s").normal1(u.Q(5.0, "s"))
    >>> jnp.sqrt(jnp.sum(U1.value**2))
    Array(1., dtype=float64)

    """

    curve: Callable[[Any], Any]
    """The constructing curve."""

    tau_unit: u.AbstractUnit | None = eqx.field(
        default=None, static=True, converter=unit_or_none
    )
    """The unit of the curve parameter tau; `None` infers it from the call."""

    station: Any = None
    """Optional fixed station along the curve (a leaf); `None` means "use tau"."""

    tau_0: u.AbstractQuantity | None = None
    """Reference parameter value where the initial frame is defined (a leaf).

    `None` is resolved to ``Q(0.0, tau_unit)`` by ``__post_init__`` when the
    unit is declared, and at call time when it is being inferred -- the
    earliest that unit is known.
    """

    initial_normal: Any = None
    """Initial U1 vector at tau_0 (dimensionless jax array, or None for auto)."""

    # `static=True` is *safe* because a `DiffEqSolver` is hashable, compares
    # equal across freshly-built instances, and contributes no **array** leaves
    # -- its ten leaves are Python floats/ints, a bool and a function, from
    # ``PIDController``.  So nothing that belongs in a buffer is frozen into
    # the treedef, and `equinox.filter_jit` retraces on a genuine tolerance
    # change but not on a fresh-but-equal solver.
    #
    # Static is *preferable* because dynamic leaks the integrator into the
    # builder's pytree: 12 leaves instead of 2, and a `jax.tree_util.tree_map`
    # meant for the curve's parameters silently rescales the tolerances and
    # doubles the step budget along with them.  ``max_steps`` is additionally
    # a `diffrax` loop bound, which has to be a concrete `int`.

    # `default_factory`, not `default`, only to keep Sphinx quiet: a plain
    # default leaves a *callable* class attribute behind, and `autodoc` then
    # tries to render a signature for it and dies ("error while formatting
    # signature ...: list assignment index out of range"), which `-W` turns
    # into a failed docs build.  The factory returns the module constant
    # itself, so every builder still shares one object.
    diffeqsolver: DiffEqSolver = eqx.field(
        default_factory=lambda: _DIFFEQSOLVER, static=True
    )
    """Solver, step-size controller, adjoint and step budget for the ODE.

    Named for its type rather than ``solver``, which is now one level down
    (``builder.diffeqsolver.solver``) and no longer means the whole solve.
    """

    def __post_init__(self) -> None:
        """Resolve a `None` ``tau_0`` to zero in a *declared* ``tau_unit``.

        Materialising it is what makes the default ``tau_0`` a pytree leaf, and
        so differentiable and vmappable exactly like an explicit one.

        An inferred ``tau_unit`` is not known until a parameter arrives, so
        there the resolution moves to `_solve_U1` and the *default* ``tau_0``
        is not a leaf. Pass it explicitly to differentiate through it -- which
        also declares its unit, and is the only way to say "start somewhere
        other than zero" regardless.
        """
        if self.tau_0 is None and self.tau_unit is not None:
            self.tau_0 = u.Q(0.0, self.tau_unit)

    # ---------------------------------------------------------------

    def _tangent_at(self, g: Any, /) -> u.AbstractQuantity:
        r"""Compute unit tangent $\mathbf{T} = \gamma'/\|\gamma'\|$."""
        # Reads the unit off `g` rather than being handed it: this is passed
        # whole to `unxt.experimental.jacfwd`, whose `units=` tuple must match
        # the positional arity, so a second parameter cannot be added without
        # a `functools.partial` at every call. `g` always arrives from
        # `_param`, so the read is a `unit_of` on a `Quantity` and cannot fail.
        dcurve = u.experimental.jacfwd(self.curve, units=(self._tau_unit_at(g),))
        return _normalize(dcurve(g.astype(float)))

    def _transport_start(self, tau_unit: Any, /) -> tuple[Any, Any, Any, Any]:
        """Resolve everything the transport ODE needs before it can run.

        Shared by `_solve_U1` and `rotation_matrices`, which differ only in how
        far they sweep: one parameter or a batch of them. ``tau_0`` is restated
        in ``tau_unit`` so the nested `_tangent_at` infers the same unit as the
        solve rather than whichever convertible one it was given in.
        """
        tau_0_in = self.tau_0
        tau_0 = u.Q(0.0 if tau_0_in is None else tau_0_in.ustrip(tau_unit), tau_unit)

        T0_val = self._tangent_at(tau_0).value
        if self.initial_normal is not None:
            # A supplied vector is NOT trusted to be unit or normal-plane: the
            # transport ODE conserves any error in it forever, so R would not
            # be a rotation.  Gram--Schmidt it exactly as the auto path does.
            U1_0_val = _orthonormalize(_float(self.initial_normal), T0_val)
        else:
            U1_0_val = _auto_initial_normal(T0_val)

        # Pre-compute dT/dtau as a callable.  This avoids nesting AD inside
        # the ODE right-hand-side, which would be both slower and harder
        # for JAX to trace.
        dTangent_fn = u.experimental.jacfwd(self._tangent_at, units=(tau_unit,))
        return tau_0, _float(tau_0.ustrip(tau_unit)), U1_0_val, dTangent_fn

    def _transport_rhs(
        self, tau_0_val: Any, span: Any, tau_unit: Any, dTangent_fn: Any, /
    ) -> Any:
        """Build the parallel-transport right-hand side over rescaled ``s``."""

        def ode_rhs(s: Any, U1_flat: Any, args: Any) -> Any:
            del args
            t_q = u.Q(tau_0_val + s * span, tau_unit)
            T_val = self._tangent_at(t_q).value
            dT_val = dTangent_fn(t_q).value
            # Project U1 onto dT, negate, then scale by T -- and by span/ds.
            return -span * jnp.dot(U1_flat, dT_val) * T_val

        return ode_rhs

    def _solve_U1(self, g: Any, tau_unit: Any, /) -> Array:
        r"""Compute $\mathbf{U}_1$ via ODE integration from $\tau_0$.

        Solves the parallel-transport ODE $d\mathbf{U}_1/d\tau = -(\mathbf{U}_1
        \cdot \mathbf{T}')\,\mathbf{T}$ with `diffrax.diffeqsolve`, over the
        rescaled parameter $s \in [0, 1]$ with $\tau(s) = \tau_0 + s\,\Delta$,
        $\Delta = \tau - \tau_0$.

        The rescaling is what makes the solve differentiable in $\tau$
        everywhere.  Integrating over $[\tau_0, \tau]$ directly puts $\tau$ in
        the integration *bound*, and at $\tau = \tau_0$ the solver loop takes
        zero steps: the value is right but $d/d\tau$ comes back as $0$ instead
        of the true $-(\mathbf{U}_{1,0} \cdot \mathbf{T}'_0)\,\mathbf{T}_0$,
        silently corrupting every tangent/jet propagation at the (very common)
        default $\tau_0 = 0$.  Over $s \in [0, 1]$ the interval is always
        unit-length and $\tau$ enters through the vector field instead, so its
        derivative survives.  Both signs of $\Delta$ are handled by the same
        expression -- no forward-only workaround is needed.
        """
        _, tau_0_val, U1_0_val, dTangent_fn = self._transport_start(tau_unit)
        dtau = _float(g.ustrip(tau_unit)) - tau_0_val
        ode_rhs = self._transport_rhs(tau_0_val, dtau, tau_unit, dTangent_fn)

        sol = self.diffeqsolver(dfx.ODETerm(ode_rhs), 0.0, 1.0, None, U1_0_val)
        U1_val = sol.ys[-1]
        # Re-normalise for numerical safety.
        return U1_val / jnp.linalg.norm(U1_val)

    def rotation_matrix(self, tau: Any, /) -> Array:
        r"""Compute the rotation $R = [T;\,U_1;\,U_2]$.

        Steps:

        1. Evaluate the tangent $\mathbf{T}$ from the first derivative.
        2. Solve the parallel-transport ODE for $\mathbf{U}_1$.
        3. Cross product: $\mathbf{U}_2 = \mathbf{T} \times \mathbf{U}_1$.
        4. Stack rows into a $3 \times 3$ matrix.

        This is the cheap way to get the triad: one ODE solve for all three
        rows.  Reach for it rather than calling `normal1` and `normal2`
        separately, which solves twice for the same information.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> R = cxfc.BishopBuilder(circle, "s").rotation_matrix(u.Q(0.0, "s"))
        >>> bool(jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6))
        True

        """
        # For a two-argument curve `tau` is the time: transport runs along
        # that time slice, at the pinned station. See `_resolve`.
        b, p = self._resolve(tau)
        g, tau_unit = b._param(p)
        T_val = b._tangent_at(g).value
        U1_val = b._solve_U1(g, tau_unit)
        U2_val = jnp.cross(T_val, U1_val)
        return jnp.stack([T_val, U1_val, U2_val])

    def rotation_matrices(self, taus: Any, /) -> Array:
        r"""Frames at many $\tau$, from a **single** ODE solve.

        `rotation_matrix` runs one solve per parameter, so evaluating $N$ of
        them costs $N$ solves. The transport was already reparametrised onto
        $s \in [0, 1]$ with $\tau$ carried in the vector field, which fixes the
        integration bounds -- so one solve with `diffrax.SaveAt` returns every
        parameter at once.

        Jitted on a helix at ``rtol=atol=1e-10``, this is ~4x faster at 16
        parameters and ~9x at 64: the one solve is amortised over the batch.

        Every $\tau$ must lie on one side of ``tau_0``. The solve marches
        outward from ``tau_0`` in one monotonic sweep, so a set straddling it
        would need two; that is refused rather than silently split, via
        `equinox.error_if` so it also fires under ``jit``.

        Parameters
        ----------
        taus
            Parameters to evaluate, as one batched `unxt.Quantity`.

        Returns
        -------
        Array
            Shape ``(*batch, 3, 3)``, matching `rotation_matrix` per element.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> b = cxfc.BishopBuilder(circle, "s")
        >>> Rs = b.rotation_matrices(u.Q(jnp.asarray([0.5, 1.0]), "s"))
        >>> Rs.shape
        (2, 3, 3)

        It agrees with the per-parameter accessor:

        >>> one = b.rotation_matrix(u.Q(1.0, "s"))
        >>> bool(jnp.allclose(Rs[1], one, atol=1e-8))
        True

        """
        # Routing first, exactly as `rotation_matrix` does via `_resolve`.
        # Skipping it silently ignored a pinned `station` and returned a frame
        # per tau where the per-tau accessor correctly returns the same one.
        if _is_two_argument(self.curve):
            raise ValueError(_MSG_BATCH_TWO_ARGUMENT)

        # Resolved here rather than taken from `_param`: this is the batched
        # entry, and `taus` is a whole array of parameters instead of the one
        # a station could stand in for. `_transport_start` then restates
        # `tau_0` in the same unit, exactly as it does for `_solve_U1`.
        tau_unit = self._tau_unit_at(taus)
        taus_val = _float(u.ustrip(tau_unit, taus))

        # Validate before anything indexes `taus_val` -- including the station
        # branch below, which used to reach for element 0 first and so met an
        # empty batch with `IndexError` rather than the message meant for it.
        # `SaveAt(ts=...)` is a 1-D grid, and `span`/`argsort` below assume the
        # same, so rank is checked here rather than surfacing as a `dot_general`
        # complaint from inside the solve.
        if taus_val.ndim != 1:
            raise ValueError(_MSG_BATCH_RANK.format(shape=taus_val.shape))
        if taus_val.size == 0:
            msg = "`rotation_matrices` needs at least one tau; got an empty batch."
            raise ValueError(msg)

        if self.station is not None:
            # `_param` pins every tau to the station, so all frames coincide;
            # one solve answers the whole batch by construction.
            one = self.rotation_matrix(u.Q(taus_val[0], tau_unit))
            return jnp.broadcast_to(one, (*taus_val.shape, 3, 3))

        _, tau_0_val, U1_0_val, dTangent_fn = self._transport_start(tau_unit)

        # A mixed sign means the sweep would have to reverse mid-solve. Guard
        # `offs` itself, so the check is what the arithmetic below depends on.
        offs = taus_val - tau_0_val
        straddles = jnp.any(offs < 0.0) & jnp.any(offs > 0.0)
        offs = eqx.error_if(offs, straddles, _MSG_STRADDLES_TAU_0)

        # The furthest parameter sets the sweep; the rest are interior points of
        # the same solve. When every tau *is* tau_0 the span is zero, and that
        # is kept rather than substituted: `t_q = tau_0 + s*0` holds the solve
        # at tau_0 and the right-hand side scales to zero, so the frame stays
        # put. Substituting a nonzero span would march the curve away from
        # tau_0 to answer a question only about tau_0 -- wrong for a curve
        # defined only near it. Only the division needs guarding.
        span = offs[jnp.argmax(jnp.abs(offs))]
        # One `where`, not two: `span` is the largest-magnitude offset, so a
        # zero span means every offset is zero and `0 / 1` is already 0.
        ss = offs / jnp.where(span == 0.0, 1.0, span)

        ode_rhs = self._transport_rhs(tau_0_val, span, tau_unit, dTangent_fn)

        # `SaveAt` requires ascending ``ts``, which the caller's order need not
        # be -- and never is when tau < tau_0, where dividing by a negative
        # span reverses it. Sort for the solve, then invert the permutation so
        # the result matches the parameters as given.
        order = jnp.argsort(ss)
        sol = self.diffeqsolver(
            dfx.ODETerm(ode_rhs),
            0.0,
            1.0,
            None,
            U1_0_val,
            saveat=dfx.SaveAt(ts=ss[order]),
        )
        U1_sorted = sol.ys / jnp.linalg.norm(sol.ys, axis=-1, keepdims=True)
        U1s = U1_sorted[jnp.argsort(order)]

        Ts = jax.vmap(lambda tv: self._tangent_at(u.Q(tv, tau_unit)).value)(taus_val)
        U2s = jnp.cross(Ts, U1s)
        return jnp.stack([Ts, U1s, U2s], axis=-2)

    # ---------------------------------------------------------------
    # Convenience accessors (location inherited from the ABC)

    def tangent(self, tau: Any, /) -> u.Q:
        r"""Return the unit tangent vector $\mathbf{T}(\tau)$ (row 0 of R).

        Overrides the base implementation, which would take row 0 of the full
        rotation matrix — and so pay for the parallel-transport ODE solve that
        only $\mathbf{U}_1$ needs. The tangent is just the normalised first
        derivative; the value is identical.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.BishopBuilder(circle, "s").tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        """
        b, p = self._resolve(tau)
        g, _ = b._param(p)
        return u.Q(b._tangent_at(g).value, "")

    def normal1(self, tau: Any, /) -> u.Q:
        r"""First parallel-transported normal $\mathbf{U}_1(\tau)$ (row 1 of R).

        This vector is obtained by solving the parallel-transport ODE from the
        reference parameter $\tau_0$.  It is perpendicular to the tangent and
        rotation-minimising: the angular velocity of the frame about the tangent
        is zero.

        Each call runs its own ODE solve.  If you want more than one row of the
        triad, take them from a single `rotation_matrix` call instead --
        ``normal1`` plus ``normal2`` costs two solves where ``rotation_matrix``
        costs one (measured ~1.7x warm).  `tangent` is the exception: it needs
        no solve at all.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> U1 = cxfc.BishopBuilder(circle, "s").normal1(u.Q(0.0, "s"))
        >>> float(jnp.linalg.norm(U1.value))
        1.0

        """
        return u.Q(self.rotation_matrix(tau)[1], "")

    def normal2(self, tau: Any, /) -> u.Q:
        r"""Second normal $\mathbf{U}_2(\tau) = \mathbf{T} \times \mathbf{U}_1$.

        The second normal completes the right-handed orthonormal triad.  It is
        computed as the cross product of the tangent and the first normal, so it
        is automatically perpendicular to both.

        Like `normal1`, this runs its own ODE solve; prefer one
        `rotation_matrix` call when you want more than one row.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> U2 = cxfc.BishopBuilder(circle, "s").normal2(u.Q(0.0, "s"))
        >>> float(jnp.linalg.norm(U2.value))
        1.0

        """
        return u.Q(self.rotation_matrix(tau)[2], "")


#####################################################################
# Frame


@final
class BishopFrame(AbstractParallelTransportFrame[FrameT]):
    """Bishop (rotation-minimising) curve-attached reference frame.

    A reference frame defined relative to a base frame by a
    `coordinax.transforms.TimeDep` wrapping a `BishopBuilder`.  At each
    parameter value ``tau``, the frame is centred at the curve position with
    axes ``(T, U1, U2)`` obtained via parallel transport.

    Unlike `FrenetSerretFrame`, this frame is well-defined even at
    zero-curvature points.

    The evolution parameter ``tau`` is **not** stored on the frame; it is
    supplied at evaluation time via ``act(op, tau, x)``.

    Parameters
    ----------
    base_frame : AbstractReferenceFrame
        The ambient reference frame.
    xop : TimeDep
        The tau-dependent rotation-minimising transform from ``base_frame`` to
        this frame.
    xop_inv : TimeDep
        Its inverse.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    Build a frame relative to Alice:

    >>> b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle, "s")
    >>> b_frame.base_frame
    Alice()

    >>> isinstance(b_frame.xop.builder, cxfc.BishopBuilder)
    True

    Get the frame transition operator and apply at tau=0:

    >>> op = cxf.frame_transition(cxf.Alice(), b_frame)
    >>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
    >>> result = cxfm.act(op, u.Q(0.0, "s"), p)
    >>> jnp.allclose(result.value, jnp.array([0., 0., 0.]), atol=1e-5)
    Array(True, dtype=bool)

    """

    base_frame: FrameT
    xop: cxfm.TimeDep
    xop_inv: cxfm.TimeDep

    @classmethod
    def from_curve(
        cls,
        base_frame: FrameT,
        curve: Callable[[Any], Any],
        /,
        tau_unit: u.AbstractUnit | str | None = None,
        *,
        station: Any = None,
        tau_0: u.AbstractQuantity | None = None,
        initial_normal: Any | None = None,
        diffeqsolver: DiffEqSolver = _DIFFEQSOLVER,
    ) -> "BishopFrame[FrameT]":
        r"""Construct a BishopFrame from a base frame and curve.

        Parameters
        ----------
        base_frame : AbstractReferenceFrame
            The ambient reference frame.
        curve : Callable
            A function ``tau -> Quantity[float, (3,)]``.
        tau_unit : AbstractUnit or str, optional
            Unit of the curve parameter for differentiation.  `None` (the
            default) reads it off the parameter the frame is evaluated at.
            There is no neutral unit to default to -- a curve parameter may be
            a time, an arc length, or an affine parameter -- so rather than
            pick one, take the one the caller already stated by passing a
            `Quantity`.
        station : optional
            A fixed station along the curve; when given the frame is a fixed
            frame *field* along the curve rather than a moving frame.
        tau_0 : Quantity, optional
            Reference parameter.  Defaults to ``Q(0.0, tau_unit)``.
        initial_normal : array-like, optional
            Dimensionless 3-vector for $\mathbf{U}_{1,0}$.
        diffeqsolver : DiffEqSolver, optional
            `diffraxtra.DiffEqSolver` configuring the parallel-transport
            solve; see `BishopBuilder`.  Defaults to the same object the
            builder does, so derive from it with `dataclasses.replace` rather
            than constructing one from scratch -- a fresh ``DiffEqSolver``
            takes `diffrax`'s defaults, `RecursiveCheckpointAdjoint` included,
            which silently costs forward mode.

        Returns
        -------
        BishopFrame

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.frames as cxf
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "km")

        >>> frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle, "s")
        >>> frame.base_frame
        Alice()

        """
        builder = BishopBuilder(
            curve, tau_unit, station, tau_0, initial_normal, diffeqsolver
        )
        xop = cxfm.TimeDep(builder)
        return cls(base_frame=base_frame, xop=xop, xop_inv=xop.inverse)
