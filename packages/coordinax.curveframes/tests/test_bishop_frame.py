"""Tests for BishopFrame and BishopTransform act dispatches."""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import unxt as u

import coordinax.curveframes as cxfc

# ===================================================================
# Test fixtures


def _circle_curve(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Unit circle in xy-plane, radius 1 km, period 2pi s."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def _straight_line(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Straight line along x-axis (kappa=0)."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")


@pytest.fixture
def circle_bishop_transform() -> cxfc.BishopTransform:
    """BishopTransform built from circle curve."""
    return cxfc.BishopTransform.from_curve(_circle_curve)


@pytest.fixture
def circle_bishop_frame() -> cxfc.BishopFrame:
    """BishopFrame relative to Alice."""
    return cxfc.BishopFrame.from_curve(cxf.Alice(), _circle_curve)


@pytest.fixture
def line_bishop_frame() -> cxfc.BishopFrame:
    """BishopFrame on a straight line, relative to Alice."""
    return cxfc.BishopFrame.from_curve(cxf.Alice(), _straight_line)


# ===================================================================
# Helpers


def _as_arr(x: object, unit: str) -> np.ndarray:
    assert isinstance(x, u.AbstractQuantity)
    return np.asarray(u.ustrip(unit, x), dtype=float)


# ===================================================================
# Group A: AbstractParallelTransportFrame structure


class TestBishopFrameStructure:
    """Verify BishopFrame class hierarchy."""

    def test_is_subclass_of_abstract_curve_frame(self) -> None:
        assert issubclass(cxfc.BishopFrame, cxfc.AbstractParallelTransportFrame)

    def test_is_subclass_of_abstract_transformed_frame(self) -> None:
        assert issubclass(cxfc.BishopFrame, cxf.AbstractTransformedReferenceFrame)


# ===================================================================
# Group B: BishopFrame creation


class TestBishopFrameCreation:
    """Verify constructors and field constraints."""

    def test_direct_construction(self, circle_bishop_transform) -> None:
        """Construct BishopFrame from base_frame + xop."""
        frame = cxfc.BishopFrame(
            base_frame=cxf.Alice(),
            xop=circle_bishop_transform,
            xop_inv=circle_bishop_transform.inverse,
        )
        assert isinstance(frame, cxfc.BishopFrame)
        assert isinstance(frame.base_frame, cxf.Alice)
        assert isinstance(frame.xop, cxfc.BishopTransform)

    def test_from_curve_constructor(self) -> None:
        """from_curve builds BishopFrame from base_frame + curve."""
        frame = cxfc.BishopFrame.from_curve(cxf.Alice(), _circle_curve)
        assert isinstance(frame, cxfc.BishopFrame)
        assert isinstance(frame.base_frame, cxf.Alice)
        assert isinstance(frame.xop, cxfc.BishopTransform)

    def test_from_curve_with_tau_unit(self) -> None:
        """from_curve accepts tau_unit parameter."""
        frame = cxfc.BishopFrame.from_curve(cxf.Alice(), _circle_curve, tau_unit="yr")
        assert frame.xop.tau_unit == u.unit("yr")

    def test_xop_is_bishop_transform(self, circle_bishop_frame) -> None:
        assert isinstance(circle_bishop_frame.xop, cxfc.BishopTransform)

    def test_is_abstract_transformed_reference_frame(self, circle_bishop_frame) -> None:
        assert isinstance(circle_bishop_frame, cxf.AbstractTransformedReferenceFrame)

    def test_straight_line_construction(self, line_bishop_frame) -> None:
        """BishopFrame works on straight line (kappa=0)."""
        assert isinstance(line_bishop_frame, cxfc.BishopFrame)


# ===================================================================
# Group C: act dispatches for BishopTransform on Quantity


class TestActBishopTransformQuantity:
    """Test act() on Quantity with BishopTransform."""

    def test_act_forward_at_tau_zero(self, circle_bishop_transform) -> None:
        """Forward transform at tau=0 on a point at gamma(0).

        p = gamma(0) => delta=0 => result = (0,0,0)
        """
        tau = u.Q(0, "s")
        p = u.Q(jnp.array([1, 0, 0]), "km")
        result = cxfm.act(circle_bishop_transform, tau, p)
        np.testing.assert_allclose(_as_arr(result, "km"), [0, 0, 0], atol=1e-5)

    def test_act_inverse_roundtrip(self, circle_bishop_transform) -> None:
        """Forward then inverse recovers original point."""
        tau = u.Q(0.5, "s")
        p = u.Q(jnp.array([3, -1, 2]), "km")

        p_curve = cxfm.act(circle_bishop_transform, tau, p)
        p_back = cxfm.act(circle_bishop_transform.inverse, tau, p_curve)

        np.testing.assert_allclose(_as_arr(p_back, "km"), _as_arr(p, "km"), atol=1e-3)

    def test_act_at_different_tau_values(self, circle_bishop_transform) -> None:
        """Different tau values give different results for same point."""
        p = u.Q(jnp.array([2, 0, 0]), "km")

        r1 = cxfm.act(circle_bishop_transform, u.Q(0, "s"), p)
        r2 = cxfm.act(circle_bishop_transform, u.Q(1, "s"), p)

        assert not np.allclose(_as_arr(r1, "km"), _as_arr(r2, "km"), atol=1e-3)


# ===================================================================
# Group D: frame_transition integration


class TestFrameTransitionBishop:
    """Test frame_transition with BishopFrame."""

    def test_transition_to_bishop_frame(self, circle_bishop_frame) -> None:
        op = cxf.frame_transition(cxf.Alice(), circle_bishop_frame)
        assert isinstance(op, cxfm.AbstractTransform)

    def test_transition_from_bishop_frame(self, circle_bishop_frame) -> None:
        op = cxf.frame_transition(circle_bishop_frame, cxf.Alice())
        assert isinstance(op, cxfm.AbstractTransform)

    def test_roundtrip_alice_to_bishop_and_back(self, circle_bishop_frame) -> None:
        """Alice -> Bishop -> Alice is identity."""
        tau = u.Q(0.5, "s")
        p = u.Q(jnp.array([3, -1, 2]), "km")

        op_fwd = cxf.frame_transition(cxf.Alice(), circle_bishop_frame)
        op_bwd = cxf.frame_transition(circle_bishop_frame, cxf.Alice())

        p_bishop = cxfm.act(op_fwd, tau, p)
        p_back = cxfm.act(op_bwd, tau, p_bishop)

        np.testing.assert_allclose(_as_arr(p_back, "km"), _as_arr(p, "km"), atol=1e-3)

    def test_alice_bishop_alex_chain(self) -> None:
        """Alice -> Bishop(tau) -> Alex chain and reverse."""
        b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), _circle_curve)
        tau = u.Q(0, "s")
        p = u.Q(jnp.array([2, 0, 0]), "km")

        # Alice -> Bishop
        op_a_to_b = cxf.frame_transition(cxf.Alice(), b_frame)
        p_bishop = cxfm.act(op_a_to_b, tau, p)

        # Bishop -> Alice
        op_b_to_a = cxf.frame_transition(b_frame, cxf.Alice())
        p_back = cxfm.act(op_b_to_a, tau, p_bishop)

        np.testing.assert_allclose(_as_arr(p_back, "km"), _as_arr(p, "km"), atol=1e-3)

        # Bishop -> Alex
        op_b_to_alex = cxf.frame_transition(b_frame, cxf.Alex())
        p_alex = cxfm.act(op_b_to_alex, tau, p_bishop)

        # Alex -> Bishop
        op_alex_to_b = cxf.frame_transition(cxf.Alex(), b_frame)
        p_bishop2 = cxfm.act(op_alex_to_b, tau, p_alex)

        np.testing.assert_allclose(
            _as_arr(p_bishop2, "km"), _as_arr(p_bishop, "km"), atol=1e-3
        )

    def test_full_chain_alice_bishop_alex_roundtrip(self) -> None:
        """Alice -> Bishop -> Alex -> Bishop -> Alice recovers original."""
        b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), _circle_curve)
        tau = u.Q(0.3, "s")
        p = u.Q(jnp.array([5, -2, 1]), "km")

        op1 = cxf.frame_transition(cxf.Alice(), b_frame)
        op2 = cxf.frame_transition(b_frame, cxf.Alex())
        p_alex = cxfm.act(op2, tau, cxfm.act(op1, tau, p))

        op3 = cxf.frame_transition(cxf.Alex(), b_frame)
        op4 = cxf.frame_transition(b_frame, cxf.Alice())
        p_back = cxfm.act(op4, tau, cxfm.act(op3, tau, p_alex))

        np.testing.assert_allclose(_as_arr(p_back, "km"), _as_arr(p, "km"), atol=1e-2)

    def test_straight_line_frame_transition(self, line_bishop_frame) -> None:
        """Frame transition works on a straight line (kappa=0)."""
        tau = u.Q(1, "s")
        p = u.Q(jnp.array([2, 1, 0]), "km")

        op = cxf.frame_transition(cxf.Alice(), line_bishop_frame)
        p_bishop = cxfm.act(op, tau, p)

        op_inv = cxf.frame_transition(line_bishop_frame, cxf.Alice())
        p_back = cxfm.act(op_inv, tau, p_bishop)

        np.testing.assert_allclose(_as_arr(p_back, "km"), _as_arr(p, "km"), atol=1e-3)


# ===================================================================
# Group E: JAX compatibility


class TestBishopFrameJAX:
    """JAX transformations work with BishopFrame transitions."""

    def test_act_jit(self, circle_bishop_transform) -> None:
        """Act with BishopTransform is JIT-compatible."""
        tau = u.Q(0, "s")
        p = u.Q(jnp.array([2, 0, 0]), "km")

        result_eager = cxfm.act(circle_bishop_transform, tau, p)
        result_jit = jax.jit(lambda t, x: cxfm.act(circle_bishop_transform, t, x))(
            tau, p
        )

        np.testing.assert_allclose(
            _as_arr(result_jit, "km"), _as_arr(result_eager, "km"), atol=1e-5
        )

    def test_act_vmap_over_tau(self, circle_bishop_transform) -> None:
        """Act can be vmapped over the tau parameter."""
        taus = u.Q(jnp.linspace(0, 2, 5), "s")
        p = u.Q(jnp.array([2, 0, 0]), "km")

        results = jax.vmap(lambda t: cxfm.act(circle_bishop_transform, t, p))(taus)

        assert results.shape == (5, 3)


# ===================================================================
# Group F: Active semantics


class TestBishopActiveSemantics:
    """Active-transform semantics hold for BishopTransform."""

    def test_forward_moves_point_to_curve_frame(self, circle_bishop_transform) -> None:
        """Point at gamma(0) maps to (0,0,0) in the curve frame."""
        tau = u.Q(0, "s")
        p_on_curve = u.Q(jnp.array([1, 0, 0]), "km")
        result = cxfm.act(circle_bishop_transform, tau, p_on_curve)

        np.testing.assert_allclose(_as_arr(result, "km"), [0, 0, 0], atol=1e-5)

    def test_inverse_moves_point_back_to_ambient(self, circle_bishop_transform) -> None:
        """Origin of curve frame at tau=0 maps back to gamma(0)."""
        tau = u.Q(0, "s")
        p_origin = u.Q(jnp.array([0, 0, 0]), "km")
        result = cxfm.act(circle_bishop_transform.inverse, tau, p_origin)

        np.testing.assert_allclose(_as_arr(result, "km"), [1, 0, 0], atol=1e-3)

    def test_frame_transition_matches_direct_transform(
        self, circle_bishop_frame, circle_bishop_transform
    ) -> None:
        """frame_transition(Alice, bishop_frame) matches direct act."""
        tau = u.Q(0.5, "s")
        p = u.Q(jnp.array([2, 1, 0]), "km")

        op = cxf.frame_transition(cxf.Alice(), circle_bishop_frame)
        result_ft = cxfm.act(op, tau, p)
        result_direct = cxfm.act(circle_bishop_transform, tau, p)

        np.testing.assert_allclose(
            _as_arr(result_ft, "km"), _as_arr(result_direct, "km"), atol=1e-5
        )
