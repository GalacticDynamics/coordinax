"""Bishop-specific behaviour: the straight line, tau_0, and the helix.

Structural guarantees shared with Frenet-Serret are asserted once in
`test_parallel_transport_contract.py`. What is left here is what Bishop does
that Frenet-Serret cannot: stay well-defined on a curve with kappa=0, where the
Frenet frame is singular -- plus the `tau_0` / `initial_normal` transport
parameters, which only a parallel-transported frame has.
"""

__all__: tuple[str, ...] = ()

import dataclasses

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffraxtra import DiffEqSolver

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc
from .conftest import circle, helix, straight_line

#: Base point and velocity for the tangent/jet tests, as component dicts.
AT = {"x": u.Q(2.0, "km"), "y": u.Q(-1.0, "km"), "z": u.Q(3.0, "km")}
VEL = {"x": u.Q(0.1, "km/s"), "y": u.Q(0.2, "km/s"), "z": u.Q(-0.3, "km/s")}

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def circle_bishop() -> cxfc.BishopBuilder:
    return cxfc.BishopBuilder(circle)


@pytest.fixture
def line_bishop() -> cxfc.BishopBuilder:
    return cxfc.BishopBuilder(straight_line)


@pytest.fixture
def helix_bishop() -> cxfc.BishopBuilder:
    return cxfc.BishopBuilder(helix)


@pytest.fixture
def line_bishop_frame() -> cxfc.BishopFrame:
    return cxfc.BishopFrame.from_curve(cxf.Alice(), straight_line)


# ── Straight line (kappa = 0) ────────────────────────────────────────


class TestBishopOnStraightLine:
    """Bishop is defined where Frenet-Serret is singular (kappa=0)."""

    def test_tangent(self, line_bishop: cxfc.BishopBuilder):
        """The tangent of a line along x is always (1,0,0)."""
        T = line_bishop.tangent(u.Q(5, "s"))
        np.testing.assert_allclose(T.value, [1, 0, 0], atol=1e-5)

    @pytest.mark.parametrize("field", ["normal1", "normal2"])
    def test_normals_are_unit_vectors(self, line_bishop: cxfc.BishopBuilder, field):
        """Bishop normals exist on a straight line, where N is undefined."""
        e = getattr(line_bishop, field)(u.Q(1, "s"))
        assert jnp.allclose(jnp.linalg.norm(e.value), 1, atol=1e-4)

    def test_normals_are_constant(self, line_bishop: cxfc.BishopBuilder):
        """Parallel transport along a line keeps the normals fixed."""
        np.testing.assert_allclose(
            line_bishop.normal1(u.Q(0, "s")).value,
            line_bishop.normal1(u.Q(5, "s")).value,
            atol=1e-4,
        )

    def test_frame_transition_roundtrip(self, line_bishop_frame: cxfc.BishopFrame):
        """Alice -> Bishop(line) -> Alice is the identity."""
        tau, p = u.Q(1, "s"), u.Q(jnp.array([2.0, 1.0, 0.0]), "km")
        fwd = cxf.frame_transition(cxf.Alice(), line_bishop_frame)
        bwd = cxf.frame_transition(line_bishop_frame, cxf.Alice())
        back = cxfm.act(bwd, tau, cxfm.act(fwd, tau, p))
        np.testing.assert_allclose(back.ustrip("km"), p.ustrip("km"), atol=1e-3)


# ── Transport parameters ─────────────────────────────────────────────


class TestBishopTau0:
    """`tau_0` and `initial_normal` set the reference of the transport."""

    def test_default_tau_0(self, circle_bishop: cxfc.BishopBuilder):
        """Default tau_0 is Q(0, tau_unit)."""
        assert jnp.allclose(circle_bishop.tau_0.value, 0)
        assert circle_bishop.tau_0.unit == u.unit("s")

    def test_tau_0_is_a_pytree_leaf(self, circle_bishop: cxfc.BishopBuilder):
        """tau_0 is a real leaf, not a static/closure constant."""
        leaves = jax.tree.leaves(circle_bishop)
        assert any(leaf is circle_bishop.tau_0.value for leaf in leaves)

    def test_custom_tau_0_still_yields_a_unit_tangent(self):
        """Shifting the transport origin does not disturb the tangent."""
        bt = cxfc.BishopBuilder(circle, tau_0=u.Q(1.0, "s"))
        T = bt.tangent(u.Q(1, "s"))
        assert jnp.allclose(jnp.linalg.norm(T.value), 1, atol=1e-5)

    def test_explicit_initial_normal_is_used(self):
        """An explicit initial_normal fixes U1 at tau_0."""
        n0 = jnp.array([0.0, 0.0, 1.0])
        bt = cxfc.BishopBuilder(circle, initial_normal=n0)
        np.testing.assert_allclose(bt.normal1(u.Q(0.0, "s")).value, n0, atol=1e-6)

    def test_backwards_transport_is_a_rotation(self):
        """Tau < tau_0 must integrate backwards, not return NaN.

        `odeint` integrates forward only, so a decreasing t_span silently
        yields NaN. With the default tau_0=0 that broke *every* negative tau.
        """
        R = cxfc.BishopBuilder(helix).rotation_matrix(u.Q(-1.5, "s"))
        assert jnp.all(jnp.isfinite(R))
        np.testing.assert_allclose(R @ R.T, jnp.eye(3), atol=1e-5)
        np.testing.assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-5)

    def test_supplied_initial_normal_is_orthonormalized(self):
        """A non-orthonormal `initial_normal` must not corrupt the frame.

        The transport ODE conserves any error in U1_0 forever, so a supplied
        vector that is not unit and normal-plane makes R not a rotation.
        """
        n0 = jnp.array([0.0, 1.0, 0.0])  # neither unit-normal to T nor unique
        bt = cxfc.BishopBuilder(helix, initial_normal=n0)
        R = bt.rotation_matrix(u.Q(1.0, "s"))
        np.testing.assert_allclose(R @ R.T, jnp.eye(3), atol=1e-5)
        np.testing.assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-5)

    @pytest.mark.parametrize(
        "n0",
        [
            [2.0, 0.0, 0.0],  # plainly parallel
            [1e-12, 0.0, 0.0],  # parallel *and* tiny: still no normal component
            [0.0, 0.0, 0.0],  # no direction at all
        ],
    )
    def test_initial_normal_parallel_to_tangent_raises(self, n0):
        """A degenerate `initial_normal` fails loudly rather than as NaN.

        The guard is on the rejection *relative* to ``|v|``, so shrinking a
        parallel vector must not sneak it past.
        """
        # Tangent of the straight line at tau_0 = 0 is +x.
        bt = cxfc.BishopBuilder(straight_line, initial_normal=jnp.array(n0))
        with pytest.raises(Exception, match="parallel to the tangent"):
            bt.rotation_matrix(u.Q(1.0, "s"))

    def test_small_initial_normal_is_a_direction_not_a_magnitude(self):
        """A valid but tiny `initial_normal` must not be rejected.

        `_orthonormalize` is homogeneous of degree zero in its input, so
        ``1e-12 * n`` and ``n`` are the *same* initial condition. An absolute
        threshold on the rejection wrongly called the scaled-down one
        degenerate.
        """
        tau = u.Q(1.0, "s")
        unit = cxfc.BishopBuilder(helix, initial_normal=jnp.array([0.0, 0.0, 1.0]))
        tiny = cxfc.BishopBuilder(helix, initial_normal=jnp.array([0.0, 0.0, 1e-12]))

        R_tiny = tiny.rotation_matrix(tau)
        np.testing.assert_allclose(R_tiny, unit.rotation_matrix(tau), atol=1e-12)
        np.testing.assert_allclose(R_tiny @ R_tiny.T, jnp.eye(3), atol=1e-10)
        np.testing.assert_allclose(jnp.linalg.det(R_tiny), 1.0, atol=1e-10)


# ── JAX ──────────────────────────────────────────────────────────────


class TestBishopJAX:
    """The ODE-based normals survive JIT."""

    def test_jit_normal1(self, circle_bishop: cxfc.BishopBuilder):
        U1 = eqx.filter_jit(circle_bishop.normal1)(u.Q(0.5, "s"))
        assert jnp.allclose(jnp.linalg.norm(U1.value), 1, atol=1e-4)


class TestBishopTangentPropagation:
    r"""The transport solve must be differentiable in **forward** mode.

    `act` on tangent data and `act_jet` prolong the point action with
    ``jax.jvp``. A `jax.custom_vjp` integrator -- which is what
    ``jax.experimental.ode.odeint`` and `diffrax`'s *default*
    `RecursiveCheckpointAdjoint` both are -- cannot be ``jvp``-ed at all, so
    the whole capability was unavailable on Bishop frames while the shared
    contract (which runs on the circle) never reached this path.

    The helix is the 3-D, non-zero-torsion case, and ``tau = 0`` is the
    degenerate ``tau == tau_0`` point where a solve over ``[tau_0, tau]``
    takes zero steps and silently reports ``d/dtau = 0``.
    """

    @pytest.mark.parametrize("tau_val", [0.0, 1.0, -0.8])
    def test_act_on_velocity_matches_finite_differences(
        self, helix_bishop: cxfc.BishopBuilder, tau_val: float
    ):
        """Slot 1 is the total tau-derivative of the point action."""
        tau = u.Q(tau_val, "s")
        op = cxfm.TimeDep(helix_bishop)

        got = cxfm.act(op, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=AT)

        h = u.Q(1e-4, "s")

        def y(t):
            x = {k: AT[k] + VEL[k] * (t - tau) for k in AT}
            return cxfm.act(op, t, x, cxc.cart3d, cxr.point)

        plus, minus = y(tau + h), y(tau - h)
        for k in AT:
            want = ((plus[k] - minus[k]) / (2 * h)).ustrip("km/s")
            np.testing.assert_allclose(
                got[k].ustrip("km/s"), want, rtol=1e-5, atol=1e-6, err_msg=k
            )

    @pytest.mark.parametrize("tau_val", [0.0, 1.0, -0.8])
    def test_tangent_axis_agrees_with_frenet_serret(self, tau_val: float):
        """Row 0 of R is the unit tangent for *both* frames, exactly.

        Bishop and Frenet-Serret differ only in the normal plane, so the
        component of the prolonged velocity along the shared tangent axis must
        agree to solver accuracy -- an oracle that needs no finite differences.
        """
        tau = u.Q(tau_val, "s")
        bishop = cxfm.TimeDep(cxfc.BishopBuilder(helix))
        frenet = cxfm.TimeDep(cxfc.FrenetSerretBuilder(helix))

        kw = {"at": AT}
        got = cxfm.act(
            bishop, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, **kw
        )
        want = cxfm.act(
            frenet, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, **kw
        )
        np.testing.assert_allclose(
            got["x"].ustrip("km/s"), want["x"].ustrip("km/s"), rtol=1e-8, atol=1e-9
        )

    def test_act_jet_agrees_with_act_on_the_velocity_slot(
        self, helix_bishop: cxfc.BishopBuilder
    ):
        """The two entry points into the prolongation must not diverge."""
        tau = u.Q(0.7, "s")
        op = cxfm.TimeDep(helix_bishop)

        jet = cxfm.act_jet(op, tau, {0: AT, 1: VEL}, cxc.cart3d)
        single = cxfm.act(
            op, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=AT
        )
        for k in AT:
            np.testing.assert_allclose(
                jet[1][k].ustrip("km/s"), single[k].ustrip("km/s"), atol=1e-10
            )


# ── Helix (3D curve) ─────────────────────────────────────────────────


class TestBishopHelix:
    """The helix exercises a curve with non-zero torsion.

    The shared contract only runs on the circle, which is planar; the helix is
    the suite's only genuinely 3-D curve, so it keeps its own orthonormality
    and roundtrip checks.
    """

    @pytest.mark.parametrize("field", ["tangent", "normal1", "normal2"])
    def test_triad_is_unit(self, helix_bishop: cxfc.BishopBuilder, field: str):
        e = getattr(helix_bishop, field)(u.Q(1, "s"))
        assert jnp.allclose(jnp.linalg.norm(e.value), 1, atol=1e-5)

    def test_orthonormality(self, helix_bishop: cxfc.BishopBuilder):
        tau = u.Q(1, "s")
        T = helix_bishop.tangent(tau).value
        U1 = helix_bishop.normal1(tau).value
        U2 = helix_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0, atol=1e-4)
        assert jnp.allclose(jnp.dot(U1, U2), 0, atol=1e-4)

    def test_roundtrip_forward_inverse(self, helix_bishop: cxfc.BishopBuilder):
        """R_inv @ (R @ (p - gamma) - gamma_inv) == p, on the helix."""
        tau = u.Q(1.0, "s")
        p = u.Q(jnp.array([2.0, -1.0, 3.0]), "km")

        g = helix_bishop.location(tau)
        T = helix_bishop.tangent(tau)
        U1 = helix_bishop.normal1(tau)
        U2 = helix_bishop.normal2(tau)
        diff = p - g
        p_fwd = qnp.stack([qnp.sum(T * diff), qnp.sum(U1 * diff), qnp.sum(U2 * diff)])

        op = cxfm.TimeDep(helix_bishop)
        np.testing.assert_allclose(
            cxfm.act(op, tau, p).ustrip("km"), p_fwd.ustrip("km"), atol=1e-3
        )
        p_rec = cxfm.act(op.inverse, tau, p_fwd)
        np.testing.assert_allclose(p_rec.ustrip("km"), p.ustrip("km"), atol=1e-3)


# ── Configurable diffrax solve ───────────────────────────────────────


class ParametricHelix(eqx.Module):
    """A helix whose radius is a differentiable array leaf."""

    radius: jax.Array

    def __call__(self, tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(
            jnp.stack([self.radius * jnp.cos(t), self.radius * jnp.sin(t), 0.3 * t]),
            "km",
        )


def _orthonormality_error(R: jax.Array) -> float:
    """Worst-case ``|R R^T - I|``, the observable the tolerances control."""
    return float(jnp.max(jnp.abs(R @ R.T - jnp.eye(3))))


#: The default solve configuration, reached the way a user reaches it. The
#: builder's field is `static=True`, so `equinox.tree_at` cannot descend into
#: it (a static field is not a leaf) -- `dataclasses.replace` is the move.
_DEFAULT_SOLVE = cxfc.BishopBuilder(helix).diffeqsolver


def _configured(**kw: object) -> cxfc.BishopBuilder:
    """A helix builder whose solve config overrides only ``kw``.

    Everything not named keeps its default -- which for ``adjoint`` is the
    whole point; see `test_partial_override_preserves_the_direct_adjoint`.
    """
    return dataclasses.replace(
        cxfc.BishopBuilder(helix),
        diffeqsolver=dataclasses.replace(_DEFAULT_SOLVE, **kw),
    )


class TestBishopSolveConfiguration:
    """The `diffrax` knobs live in one builder field, and they take effect.

    They were module-level constants, so the defaults are pinned against
    measured values rather than re-derived -- a change in any of them has to
    be a deliberate edit to this file.
    """

    def test_defaults_are_the_previous_constants(self):
        """The defaults reproduce the module-level constants they replaced."""
        solve = cxfc.BishopBuilder(helix).diffeqsolver
        assert isinstance(solve, DiffEqSolver)
        assert solve.solver == dfx.Tsit5()
        assert solve.adjoint == dfx.DirectAdjoint()
        assert solve.stepsize_controller == dfx.PIDController(rtol=1e-10, atol=1e-10)
        assert solve.max_steps == 16384

    @pytest.mark.parametrize("tau_val", [0.0, 1.0, -1.5, 7.0])
    def test_default_accuracy_oracle(self, tau_val: float):
        """Pinned values: the default solve is orthonormal to ~1e-11.

        Measured at ``float64`` on the 0.3-pitch helix; the same solve is
        accurate to 9.403e-12 out at ``|tau| = 60``, where it also stays
        inside the 16384-step budget (~20 steps per unit of ``|dtau|``).
        """
        R = cxfc.BishopBuilder(helix).rotation_matrix(u.Q(tau_val, "s"))
        assert _orthonormality_error(R) < 1e-11
        np.testing.assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-11)

    def test_partial_override_preserves_the_direct_adjoint(self):
        """`dataclasses.replace` keeps every knob it is not told to change.

        This is the ergonomic that makes one aggregate field safe. A
        `DiffEqSolver` built from scratch takes `diffrax`'s own defaults --
        including `RecursiveCheckpointAdjoint`, which silently kills forward
        mode. Deriving from the default cannot do that, and the derived
        builder still does tangent propagation.
        """
        bt = _configured(stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6))
        assert bt.diffeqsolver.adjoint == dfx.DirectAdjoint()
        assert bt.diffeqsolver.solver == dfx.Tsit5()
        assert bt.diffeqsolver.max_steps == 16384
        assert bt.diffeqsolver.stepsize_controller.rtol == 1e-6

        # The contrast: forgetting the adjoint when building from scratch.
        scratch = DiffEqSolver(
            solver=dfx.Tsit5(),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=16384,
        )
        assert not isinstance(scratch.adjoint, dfx.DirectAdjoint)

        # ...and forward mode survives the derived one.
        op = cxfm.TimeDep(bt)
        tau = u.Q(0.7, "s")
        cxfm.act(op, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=AT)

    def test_the_solve_config_is_static(self):
        """Static, so the builder's pytree stays about the curve.

        A `DiffEqSolver` is hashable and contributes no *array* leaves, so
        nothing that belongs in a buffer is frozen into the treedef. Left
        dynamic, `jax.tree_util.tree_map` over the builder would rescale the
        tolerances and the step budget along with the curve parameters.
        """
        bt = _configured(
            max_steps=999,
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
        )
        # Configuring the solve adds no leaves: the `DiffEqSolver` carries ten
        # of its own (floats, ints, a bool, a function) once it is dynamic.
        assert len(jax.tree.leaves(bt)) == len(
            jax.tree.leaves(cxfc.BishopBuilder(helix))
        )

        # So a tree_map over the curve parameters cannot reach the config.
        doubled = jax.tree.map(lambda x: 2 * x if eqx.is_array(x) else x, bt)
        assert doubled.diffeqsolver.max_steps == 999
        assert doubled.diffeqsolver.stepsize_controller.rtol == 1e-3

    def test_loose_tolerances_are_measurably_worse(self):
        """A deliberately loose controller degrades orthonormality by ~1e9.

        The assertion is two-sided: the loose solve must be *far* worse than
        the default, which is what proves the field is read rather than
        merely accepted.
        """
        tau = u.Q(7.0, "s")
        default = cxfc.BishopBuilder(helix).rotation_matrix(tau)
        loose = _configured(
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3)
        ).rotation_matrix(tau)

        assert _orthonormality_error(default) < 1e-11
        assert _orthonormality_error(loose) > 1e-5
        # ...and the frames genuinely differ, not just their error estimates.
        assert float(jnp.max(jnp.abs(loose - default))) > 1e-4

    def test_alternative_solver_agrees_with_the_default(self):
        """A different solver is a different integrator, not a different answer."""
        tau = u.Q(7.0, "s")
        default = cxfc.BishopBuilder(helix).rotation_matrix(tau)
        dopri = _configured(solver=dfx.Dopri5()).rotation_matrix(tau)
        np.testing.assert_allclose(dopri, default, atol=1e-9)

    def test_max_steps_is_a_real_budget(self):
        """Too small a budget raises rather than silently truncating."""
        bt = _configured(max_steps=4)
        with pytest.raises(Exception, match="maximum number of solver steps"):
            bt.rotation_matrix(u.Q(7.0, "s"))

    def test_customised_builder_survives_jit_grad_and_vmap(self):
        """A non-default config keeps the builder a working pytree."""
        tau = u.Q(0.7, "s")
        solve = dataclasses.replace(
            _DEFAULT_SOLVE,
            solver=dfx.Dopri5(),
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8),
        )

        def build(r: jax.Array) -> cxfc.BishopBuilder:
            return cxfc.BishopBuilder(ParametricHelix(r), diffeqsolver=solve)

        one = jnp.asarray(1.0)

        # jit
        eager = build(one).rotation_matrix(tau)
        jitted = eqx.filter_jit(lambda b, t: b.rotation_matrix(t))(build(one), tau)
        np.testing.assert_allclose(jitted, eager, atol=1e-9)

        # grad w.r.t. a curve parameter, against central differences
        got = eqx.filter_grad(lambda r: build(r).rotation_matrix(tau)[1, 2])(one)
        h = 1e-5
        want = (
            build(one + h).rotation_matrix(tau)[1, 2]
            - build(one - h).rotation_matrix(tau)[1, 2]
        ) / (2 * h)
        np.testing.assert_allclose(got, want, rtol=1e-6)

        # vmap over a batch of builders sharing the static config
        radii = jnp.asarray([0.5, 1.0, 2.0])
        batched = eqx.filter_vmap(lambda r: build(r).rotation_matrix(tau))(radii)
        assert batched.shape == (3, 3, 3)
        for i, r in enumerate(radii):
            np.testing.assert_allclose(
                batched[i], build(r).rotation_matrix(tau), atol=1e-9
            )

    def test_tangent_act_works_with_the_default_adjoint(self):
        """`DirectAdjoint` still does forward mode on a customised builder.

        This is the capability the default exists to protect: it is why
        `RecursiveCheckpointAdjoint` -- `diffrax`'s own default, and so
        `DiffEqSolver`'s -- is *not* this builder's default.
        """
        tau = u.Q(0.7, "s")
        op = cxfm.TimeDep(_configured(solver=dfx.Dopri5()))

        got = cxfm.act(op, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=AT)
        jet = cxfm.act_jet(op, tau, {0: AT, 1: VEL}, cxc.cart3d)
        for k in AT:
            np.testing.assert_allclose(
                jet[1][k].ustrip("km/s"), got[k].ustrip("km/s"), atol=1e-10
            )

    def test_recursive_checkpoint_adjoint_trades_forward_mode_for_speed(self):
        """The documented trade-off, asserted so the docstring cannot rot.

        `RecursiveCheckpointAdjoint` is a `jax.custom_vjp`: reverse mode keeps
        working (and agrees with the default), but forward mode -- and so the
        whole tangent/jet capability -- is gone.
        """
        tau = u.Q(0.7, "s")
        rca = dfx.RecursiveCheckpointAdjoint()

        with pytest.raises(TypeError, match="custom_vjp"):
            cxfm.act(
                cxfm.TimeDep(_configured(adjoint=rca)),
                tau,
                VEL,
                cxc.cart3d,
                cxr.tangent_geom,
                cxr.coord_vel,
                at=AT,
            )

        # Reverse mode is unaffected, and matches the default adjoint.
        def dR(adjoint: dfx.AbstractAdjoint) -> jax.Array:
            return eqx.filter_grad(
                lambda r: cxfc.BishopBuilder(
                    ParametricHelix(r),
                    diffeqsolver=dataclasses.replace(_DEFAULT_SOLVE, adjoint=adjoint),
                ).rotation_matrix(tau)[1, 2]
            )(jnp.asarray(1.0))

        np.testing.assert_allclose(dR(rca), dR(dfx.DirectAdjoint()), rtol=1e-9)

    def test_backsolve_adjoint_is_unusable_in_both_modes(self):
        """`BacksolveAdjoint` drops *both* modes, not just reverse.

        The docstring table used to claim ``forward: yes`` for it. It is
        ``no``/``no``: the reparametrised right-hand side closes over ``dtau``
        and ``tau_0_val``, so `BacksolveAdjoint`'s backwards solve raises
        JAX's ``CustomVJPException`` ("...with respect to a closed-over
        value") whichever way it is differentiated. Nothing else in the suite
        instantiates it, so without this the table can rot again.
        """
        tau = u.Q(0.7, "s")
        bs = dfx.BacksolveAdjoint()

        # Forward: `act` on tangent data.
        with pytest.raises(Exception, match="closed-over value"):
            cxfm.act(
                cxfm.TimeDep(_configured(adjoint=bs)),
                tau,
                VEL,
                cxc.cart3d,
                cxr.tangent_geom,
                cxr.coord_vel,
                at=AT,
            )

        # Reverse: `grad` w.r.t. a curve parameter.
        with pytest.raises(Exception, match="closed-over value"):
            eqx.filter_grad(
                lambda r: cxfc.BishopBuilder(
                    ParametricHelix(r),
                    diffeqsolver=dataclasses.replace(_DEFAULT_SOLVE, adjoint=bs),
                ).rotation_matrix(tau)[1, 2]
            )(jnp.asarray(1.0))

        # ...and it is the reparametrisation, not the curve: a bare-function
        # curve carries no array leaves at all and fails identically.
        with pytest.raises(Exception, match="closed-over value"):
            eqx.filter_grad(
                lambda t: _configured(adjoint=bs).rotation_matrix(u.Q(t, "s"))[1, 2]
            )(jnp.asarray(0.7))

    def test_from_curve_forwards_the_diffeqsolver(self):
        """The frame constructor reaches the solve config, not just the builder.

        `BishopFrame.from_curve` forwards every other builder field, so a
        ``diffeqsolver`` it dropped would leave the documented entry point
        silently stuck on the default. Asserted by *effect*: the same loose
        controller that degrades accuracy on the builder must degrade it here.
        """
        tau = u.Q(7.0, "s")
        loose = dataclasses.replace(
            _DEFAULT_SOLVE, stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3)
        )
        frame = cxfc.BishopFrame.from_curve(cxf.Alice(), helix, diffeqsolver=loose)

        assert frame.xop.builder.diffeqsolver is loose
        assert _orthonormality_error(frame.xop.builder.rotation_matrix(tau)) > 1e-5

        # The default path is untouched.
        default = cxfc.BishopFrame.from_curve(cxf.Alice(), helix)
        assert default.xop.builder.diffeqsolver == _DEFAULT_SOLVE
        assert _orthonormality_error(default.xop.builder.rotation_matrix(tau)) < 1e-11
