"""Bishop-specific behaviour: the straight line, tau_0, and opaque units.

Structural guarantees shared with Frenet-Serret are asserted once in
`test_parallel_transport_contract.py`. What is left here is what Bishop does
that Frenet-Serret cannot: stay well-defined on a curve with kappa=0, where the
Frenet frame is singular -- plus the `tau_0` reference parameter, which only a
parallel-transported frame has.
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc


class TestBishopLocation:
    """The location field is the curve object itself, not a copy."""

    def test_location_is_curve(self, circle_bishop: cxfc.BishopTransform):
        assert circle_bishop.location is circle_bishop.curve


class TestBishopOnStraightLine:
    """Bishop is defined where Frenet-Serret is singular (kappa=0)."""

    def test_tangent(self, line_bishop: cxfc.BishopTransform):
        """The tangent of a line along x is always (1,0,0)."""
        T = line_bishop.tangent(u.Q(5, "s"))
        np.testing.assert_allclose(T.value, [1, 0, 0], atol=1e-5)

    @pytest.mark.parametrize("field", ["normal1", "normal2"])
    def test_normals_are_unit_vectors(
        self, line_bishop: cxfc.BishopTransform, field: str
    ):
        """Bishop normals exist on a straight line, where N is undefined."""
        e = getattr(line_bishop, field)(u.Q(1, "s"))
        assert jnp.allclose(jnp.linalg.norm(e.value), 1, atol=1e-4)

    def test_normals_are_constant(self, line_bishop: cxfc.BishopTransform):
        """Parallel transport along a line keeps the normals fixed."""
        np.testing.assert_allclose(
            line_bishop.normal1(u.Q(0, "s")).value,
            line_bishop.normal1(u.Q(5, "s")).value,
            atol=1e-4,
        )

    def test_frame_construction(self, line_bishop_frame: cxfc.BishopFrame):
        assert isinstance(line_bishop_frame, cxfc.BishopFrame)

    def test_frame_transition_roundtrip(self, line_bishop_frame: cxfc.BishopFrame, arr):
        """Alice -> Bishop(line) -> Alice is the identity."""
        tau, p = u.Q(1, "s"), u.Q(jnp.array([2, 1, 0]), "km")
        fwd = cxf.frame_transition(cxf.Alice(), line_bishop_frame)
        bwd = cxf.frame_transition(line_bishop_frame, cxf.Alice())
        back = cxfm.act(bwd, tau, cxfm.act(fwd, tau, p))
        np.testing.assert_allclose(arr(back, "km"), arr(p, "km"), atol=1e-3)


class TestBishopTau0:
    """`tau_0` sets the reference parameter for the transport."""

    def test_default_is_zero(self, circle_bishop: cxfc.BishopTransform):
        assert jnp.allclose(circle_bishop.tau_0.value, 0)

    def test_custom_tau_0_still_yields_a_unit_tangent(self, curve):
        """Shifting the transport origin does not disturb the tangent."""
        bt = cxfc.BishopTransform.from_curve(curve, tau_0=u.Q(1, "s"))
        T = bt.tangent(u.Q(1, "s"))
        assert jnp.allclose(jnp.linalg.norm(T.value), 1, atol=1e-5)

    def test_initial_normal_is_stored(self, circle_bishop: cxfc.BishopTransform):
        """`initial_normal` is None (auto) or a 3-vector, for reconstruction."""
        if circle_bishop.initial_normal is not None:
            assert circle_bishop.initial_normal.shape == (3,)


class TestBishopOpaqueUnits:
    """A curve whose internal unit (yr) differs from the caller's."""

    def test_tangent_at_zero(self, circle_yr_bishop: cxfc.BishopTransform):
        T = circle_yr_bishop.tangent(u.Q(0, "yr"))
        np.testing.assert_allclose(T.value, [0, 1, 0], atol=1e-5)

    def test_triad_orthogonal_at_zero(self, circle_yr_bishop: cxfc.BishopTransform):
        tau = u.Q(0, "yr")
        T = circle_yr_bishop.tangent(tau).value
        U1 = circle_yr_bishop.normal1(tau).value
        U2 = circle_yr_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0, atol=1e-4)
        assert jnp.allclose(jnp.dot(U1, U2), 0, atol=1e-4)

    def test_inverse_location_is_defined(self, circle_yr_bishop: cxfc.BishopTransform):
        assert circle_yr_bishop.inverse.location(u.Q(0, "yr")).shape == (3,)


class TestBishopJAX:
    """The ODE-based normals survive JIT."""

    def test_jit_normal1(self, circle_bishop: cxfc.BishopTransform):
        U1 = jax.jit(circle_bishop.normal1)(u.Q(0.5, "s"))
        assert jnp.allclose(jnp.linalg.norm(U1.value), 1, atol=1e-4)


class TestBishopHelix:
    """The helix exercises a curve with non-zero torsion.

    The shared contract only runs on the circle, which is planar; the helix is
    the suite's only genuinely 3-D curve, so it keeps its own orthonormality
    and roundtrip checks.
    """

    @pytest.mark.parametrize("field", ["tangent", "normal1", "normal2"])
    def test_triad_is_unit(self, helix_bishop: cxfc.BishopTransform, field: str):
        e = getattr(helix_bishop, field)(u.Q(1, "s"))
        assert jnp.allclose(jnp.linalg.norm(e.value), 1, atol=1e-5)

    def test_orthonormality(self, helix_bishop: cxfc.BishopTransform):
        tau = u.Q(1, "s")
        T = helix_bishop.tangent(tau).value
        U1 = helix_bishop.normal1(tau).value
        U2 = helix_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0, atol=1e-4)
        assert jnp.allclose(jnp.dot(U1, U2), 0, atol=1e-4)

    def test_roundtrip_forward_inverse(self, helix_bishop: cxfc.BishopTransform):
        """R_inv @ (R @ (p - gamma) - gamma_inv) == p, on the helix."""
        tau = u.Q(1, "s")
        p = u.Q(jnp.array([2, -1, 3]), "km")

        fields = ("tangent", "normal1", "normal2")
        diff = p - helix_bishop.location(tau)
        p_fwd = qnp.stack(
            [qnp.sum(getattr(helix_bishop, f)(tau) * diff) for f in fields]
        )

        inv = helix_bishop.inverse
        diff_inv = p_fwd - inv.location(tau)
        p_rec = qnp.stack([qnp.sum(getattr(inv, f)(tau) * diff_inv) for f in fields])

        assert jnp.allclose(p_rec.value, p.value, atol=1e-3)
