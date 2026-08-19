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
from typing import Any, cast, final

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from diffraxtra import DiffEqSolver

import coordinax.transforms as cxfm
import unxt as u

from .base import AbstractCurveFrameBuilder, AbstractParallelTransportFrame, FrameT
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

_MSG_PARALLEL_NORMAL = (
    "`initial_normal` is parallel to the tangent at tau_0; it has no component "
    "in the normal plane. Pass a vector that is not along the tangent, or leave "
    "it `None` to have one chosen automatically."
)


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
    # `<=`, not `<`, so that an all-zero `v` (threshold 0) still raises.
    w = eqx.error_if(w, norm <= 1e-12 * jnp.linalg.norm(v), _MSG_PARALLEL_NORMAL)
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
        Unit of the curve parameter.  Defaults to ``"s"``.
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

    >>> bt = cxfc.BishopBuilder(helix)
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

    The Bishop frame works on a straight line (where Frenet-Serret is singular):

    >>> def line(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([t, jnp.zeros_like(t),
    ...                           jnp.zeros_like(t)]), "m")

    >>> U1 = cxfc.BishopBuilder(line).normal1(u.Q(5.0, "s"))
    >>> jnp.sqrt(jnp.sum(U1.value**2))
    Array(1., dtype=float64)

    """

    curve: Callable[[Any], Any]
    """The constructing curve."""

    tau_unit: u.AbstractUnit = eqx.field(  # ty: ignore[invalid-assignment]
        default=u.unit("s"), static=True, converter=u.unit
    )
    """The unit of the curve parameter tau."""

    station: Any = None
    """Optional fixed station along the curve (a leaf); `None` means "use tau"."""

    tau_0: u.AbstractQuantity | None = None
    """Reference parameter value where the initial frame is defined (a leaf).

    `None` is resolved to ``Q(0.0, tau_unit)`` by ``__post_init__``.
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
        """Resolve a `None` ``tau_0`` to zero in ``tau_unit`` (a pytree leaf)."""
        if self.tau_0 is None:
            self.tau_0 = u.Q(0.0, self.tau_unit)

    # ---------------------------------------------------------------

    def _tangent_at(self, g: Any, /) -> u.AbstractQuantity:
        r"""Compute unit tangent $\mathbf{T} = \gamma'/\|\gamma'\|$."""
        dcurve = u.experimental.jacfwd(self.curve, units=(self.tau_unit,))
        return _normalize(dcurve(g.astype(float)))

    def _solve_U1(self, g: Any, /) -> Array:
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
        tau_unit = self.tau_unit
        tau_0 = cast("u.AbstractQuantity", self.tau_0)

        T0_val = self._tangent_at(tau_0).value
        if self.initial_normal is not None:
            # A supplied vector is NOT trusted to be unit or normal-plane: the
            # transport ODE conserves any error in it forever, so R would not
            # be a rotation.  Gram--Schmidt it exactly as the auto path does.
            U1_0_val = _orthonormalize(
                jnp.asarray(self.initial_normal, dtype=float), T0_val
            )
        else:
            U1_0_val = _auto_initial_normal(T0_val)

        # Pre-compute dT/dtau as a callable.  This avoids nesting AD inside
        # the ODE right-hand-side, which would be both slower and harder
        # for JAX to trace.
        dTangent_fn = u.experimental.jacfwd(self._tangent_at, units=(tau_unit,))

        tau_val = jnp.asarray(g.ustrip(tau_unit), dtype=float)
        tau_0_val = jnp.asarray(tau_0.ustrip(tau_unit), dtype=float)

        dtau = tau_val - tau_0_val

        def ode_rhs(s: Any, U1_flat: Any, args: Any) -> Any:
            """Right-hand side in the rescaled parameter ``s``."""
            del args
            t_q = u.Q(tau_0_val + s * dtau, tau_unit)
            T_val = self._tangent_at(t_q).value
            dT_val = dTangent_fn(t_q).value
            # Project U1 onto dT, negate, then scale by T -- and by dtau/ds.
            return -dtau * jnp.dot(U1_flat, dT_val) * T_val

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

        >>> R = cxfc.BishopBuilder(circle).rotation_matrix(u.Q(0.0, "s"))
        >>> bool(jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6))
        True

        """
        # For a two-argument curve `tau` is the time: transport runs along
        # that time slice, at the pinned station. See `_resolve`.
        b, p = self._resolve(tau)
        g = b._param(p)
        T_val = b._tangent_at(g).value
        U1_val = b._solve_U1(g)
        U2_val = jnp.cross(T_val, U1_val)
        return jnp.stack([T_val, U1_val, U2_val])

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

        >>> cxfc.BishopBuilder(circle).tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        """
        b, p = self._resolve(tau)
        return u.Q(b._tangent_at(b._param(p)).value, "")

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

        >>> U1 = cxfc.BishopBuilder(circle).normal1(u.Q(0.0, "s"))
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

        >>> U2 = cxfc.BishopBuilder(circle).normal2(u.Q(0.0, "s"))
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

    >>> b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle)
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
        tau_unit: u.AbstractUnit | str = "s",
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
        tau_unit : str, optional
            Unit of the curve parameter for differentiation.
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

        >>> frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle)
        >>> frame.base_frame
        Alice()

        """
        builder = BishopBuilder(
            curve, tau_unit, station, tau_0, initial_normal, diffeqsolver
        )
        xop = cxfm.TimeDep(builder)
        return cls(base_frame=base_frame, xop=xop, xop_inv=xop.inverse)
