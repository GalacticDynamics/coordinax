"""Bishop-specific behaviour: the straight line, tau_0, and the helix.

Structural guarantees shared with Frenet-Serret are asserted once in
`test_parallel_transport_contract.py`. What is left here is what Bishop does
that Frenet-Serret cannot: stay well-defined on a curve with kappa=0, where the
Frenet frame is singular -- plus the `tau_0` / `initial_normal` transport
parameters, which only a parallel-transported frame has.
"""

__all__: tuple[str, ...] = ()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc
from .conftest import circle, helix, straight_line

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

    def test_initial_normal_parallel_to_tangent_raises(self):
        """A degenerate `initial_normal` fails loudly rather than as NaN."""
        # Tangent of the straight line at tau_0 = 0 is +x.
        bt = cxfc.BishopBuilder(
            straight_line, initial_normal=jnp.array([2.0, 0.0, 0.0])
        )
        with pytest.raises(Exception, match="parallel to the tangent"):
            bt.rotation_matrix(u.Q(1.0, "s"))


# ── JAX ──────────────────────────────────────────────────────────────


class TestBishopJAX:
    """The ODE-based normals survive JIT."""

    def test_jit_normal1(self, circle_bishop: cxfc.BishopBuilder):
        U1 = eqx.filter_jit(circle_bishop.normal1)(u.Q(0.5, "s"))
        assert jnp.allclose(jnp.linalg.norm(U1.value), 1, atol=1e-4)


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
