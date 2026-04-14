"""Tests for FrenetSerretFrame and FrenetSerretTransform act dispatches."""

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


@pytest.fixture
def circle_fs_transform() -> cxfc.FrenetSerretTransform:
    """FrenetSerretTransform built from circle curve."""
    return cxfc.FrenetSerretTransform.from_curve(_circle_curve)


@pytest.fixture
def circle_fs_frame() -> cxfc.FrenetSerretFrame:
    """FrenetSerretFrame relative to Alice."""
    return cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), _circle_curve)


# ===================================================================
# Group A: AbstractParallelTransportFrame structure


class TestAbstractParallelTransportFrameStructure:
    """Verify AbstractParallelTransportFrame class hierarchy."""

    def test_is_subclass_of_abstract_transformed_frame(self) -> None:
        """AbstractParallelTransportFrame inherits AbstractTransformedReferenceFrame."""
        assert issubclass(
            cxfc.AbstractParallelTransportFrame, cxf.AbstractTransformedReferenceFrame
        )

    def test_frenet_frame_is_subclass(self) -> None:
        """FrenetSerretFrame is a subclass of AbstractParallelTransportFrame."""
        assert issubclass(cxfc.FrenetSerretFrame, cxfc.AbstractParallelTransportFrame)


# ===================================================================
# Group B: FrenetSerretFrame creation


class TestFrenetSerretFrameCreation:
    """Verify constructors and field constraints."""

    def test_direct_construction(self, circle_fs_transform) -> None:
        """Construct FrenetSerretFrame from base_frame + xop."""
        frame = cxfc.FrenetSerretFrame(
            base_frame=cxf.Alice(),
            xop=circle_fs_transform,
            xop_inv=circle_fs_transform.inverse,
        )
        assert isinstance(frame, cxfc.FrenetSerretFrame)
        assert isinstance(frame.base_frame, cxf.Alice)
        assert isinstance(frame.xop, cxfc.FrenetSerretTransform)

    def test_from_curve_constructor(self) -> None:
        """from_curve builds FrenetSerretFrame from base_frame + curve."""
        frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), _circle_curve)
        assert isinstance(frame, cxfc.FrenetSerretFrame)
        assert isinstance(frame.base_frame, cxf.Alice)
        assert isinstance(frame.xop, cxfc.FrenetSerretTransform)

    def test_from_curve_with_tau_unit(self) -> None:
        """from_curve accepts tau_unit parameter."""
        frame = cxfc.FrenetSerretFrame.from_curve(
            cxf.Alice(), _circle_curve, tau_unit="yr"
        )
        assert frame.xop.tau_unit == u.unit("yr")

    def test_xop_is_frenet_serret_transform(self, circle_fs_frame) -> None:
        """The xop field is a FrenetSerretTransform."""
        assert isinstance(circle_fs_frame.xop, cxfc.FrenetSerretTransform)

    def test_is_abstract_transformed_reference_frame(self, circle_fs_frame) -> None:
        """FrenetSerretFrame is an AbstractTransformedReferenceFrame."""
        assert isinstance(circle_fs_frame, cxf.AbstractTransformedReferenceFrame)


# ===================================================================
# Group C: act dispatches for FrenetSerretTransform on Quantity


def _as_array(x: object, unit: str) -> np.ndarray:
    assert isinstance(x, u.AbstractQuantity)
    return np.asarray(u.ustrip(unit, x), dtype=float)


class TestActFrenetSerretTransformQuantity:
    """Test act() on Quantity with FrenetSerretTransform."""

    def test_act_forward_at_tau_zero(self, circle_fs_transform) -> None:
        """Forward transform at tau=0 on a known point.

        At tau=0: gamma=(1,0,0) km, T=(0,1,0), N=(-1,0,0), B=(0,0,1)
        R @ (p - gamma) where p=(1,0,0) km => R @ (0,0,0) = (0,0,0) km.
        """
        tau = u.Q(0.0, "s")
        p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
        result = cxfm.act(circle_fs_transform, tau, p)
        np.testing.assert_allclose(_as_array(result, "km"), [0.0, 0.0, 0.0], atol=1e-6)

    def test_act_forward_off_curve(self, circle_fs_transform) -> None:
        """Forward transform at tau=0 on a point offset from the curve.

        At tau=0: gamma=(1,0,0), T=(0,1,0), N=(-1,0,0), B=(0,0,1)
        p=(2,0,0) km => delta=(1,0,0)
        R @ delta = [T·delta, N·delta, B·delta] = [0, -1, 0] km
        """
        tau = u.Q(0.0, "s")
        p = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")
        result = cxfm.act(circle_fs_transform, tau, p)
        np.testing.assert_allclose(_as_array(result, "km"), [0.0, -1.0, 0.0], atol=1e-6)

    def test_act_inverse_roundtrip(self, circle_fs_transform) -> None:
        """Forward then inverse recovers original point."""
        tau = u.Q(0.5, "s")
        p = u.Q(jnp.array([3.0, -1.0, 2.0]), "km")

        p_curve = cxfm.act(circle_fs_transform, tau, p)
        p_back = cxfm.act(circle_fs_transform.inverse, tau, p_curve)

        np.testing.assert_allclose(
            _as_array(p_back, "km"), _as_array(p, "km"), atol=1e-6
        )

    def test_act_at_different_tau_values(self, circle_fs_transform) -> None:
        """Different tau values give different results for same point."""
        p = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")

        r1 = cxfm.act(circle_fs_transform, u.Q(0.0, "s"), p)
        r2 = cxfm.act(circle_fs_transform, u.Q(1.0, "s"), p)

        # These should be different since the frame rotates
        assert not np.allclose(_as_array(r1, "km"), _as_array(r2, "km"), atol=1e-3)


# ===================================================================
# Group D: frame_transition integration


class TestFrameTransitionFrenetSerret:
    """Test frame_transition with FrenetSerretFrame."""

    def test_transition_to_fs_frame(self, circle_fs_frame) -> None:
        """frame_transition(Alice, fs_frame) returns an AbstractTransform."""
        op = cxf.frame_transition(cxf.Alice(), circle_fs_frame)
        assert isinstance(op, cxfm.AbstractTransform)

    def test_transition_from_fs_frame(self, circle_fs_frame) -> None:
        """frame_transition(fs_frame, Alice) returns an AbstractTransform."""
        op = cxf.frame_transition(circle_fs_frame, cxf.Alice())
        assert isinstance(op, cxfm.AbstractTransform)

    def test_roundtrip_alice_to_fs_and_back(self, circle_fs_frame) -> None:
        """Alice -> FS -> Alice is identity for any point."""
        tau = u.Q(0.5, "s")
        p = u.Q(jnp.array([3.0, -1.0, 2.0]), "km")

        op_fwd = cxf.frame_transition(cxf.Alice(), circle_fs_frame)
        op_bwd = cxf.frame_transition(circle_fs_frame, cxf.Alice())

        p_fs = cxfm.act(op_fwd, tau, p)
        p_back = cxfm.act(op_bwd, tau, p_fs)

        np.testing.assert_allclose(
            _as_array(p_back, "km"), _as_array(p, "km"), atol=1e-6
        )

    def test_alice_fs_alex_chain(self) -> None:
        """Alice -> FS(tau) -> Alex chain and reverse.

        We build two FrenetSerretFrames, one relative to Alice and one
        relative to Alex. Transitioning Alice -> FS_alice(tau) should
        give the same result as the direct transform followed by the
        Alice-to-Alex transition.
        """
        fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), _circle_curve)
        tau = u.Q(0.0, "s")
        p = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")

        # Alice -> FS
        op_a_to_fs = cxf.frame_transition(cxf.Alice(), fs_frame)
        p_fs = cxfm.act(op_a_to_fs, tau, p)

        # FS -> Alice
        op_fs_to_a = cxf.frame_transition(fs_frame, cxf.Alice())
        p_back = cxfm.act(op_fs_to_a, tau, p_fs)

        np.testing.assert_allclose(
            _as_array(p_back, "km"), _as_array(p, "km"), atol=1e-6
        )

        # Now test FS -> Alex: this is xop.inverse | (Alice->Alex)
        op_fs_to_alex = cxf.frame_transition(fs_frame, cxf.Alex())
        p_alex = cxfm.act(op_fs_to_alex, tau, p_fs)

        # And back: Alex -> FS
        op_alex_to_fs = cxf.frame_transition(cxf.Alex(), fs_frame)
        p_fs2 = cxfm.act(op_alex_to_fs, tau, p_alex)

        np.testing.assert_allclose(
            _as_array(p_fs2, "km"), _as_array(p_fs, "km"), atol=1e-6
        )

    def test_full_chain_alice_fs_alex_roundtrip(self) -> None:
        """Alice -> FS -> Alex -> FS -> Alice recovers original point."""
        fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), _circle_curve)
        tau = u.Q(0.3, "s")
        p = u.Q(jnp.array([5.0, -2.0, 1.0]), "km")

        # Alice -> FS -> Alex
        op1 = cxf.frame_transition(cxf.Alice(), fs_frame)
        op2 = cxf.frame_transition(fs_frame, cxf.Alex())
        p_alex = cxfm.act(op2, tau, cxfm.act(op1, tau, p))

        # Alex -> FS -> Alice
        op3 = cxf.frame_transition(cxf.Alex(), fs_frame)
        op4 = cxf.frame_transition(fs_frame, cxf.Alice())
        p_back = cxfm.act(op4, tau, cxfm.act(op3, tau, p_alex))

        np.testing.assert_allclose(
            _as_array(p_back, "km"), _as_array(p, "km"), atol=1e-5
        )


# ===================================================================
# Group E: JAX compatibility


class TestFrenetSerretFrameJAX:
    """JAX transformations work with FrenetSerretFrame transitions."""

    def test_act_jit(self, circle_fs_transform) -> None:
        """Act with FrenetSerretTransform is JIT-compatible."""
        tau = u.Q(0.0, "s")
        p = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")

        result_eager = cxfm.act(circle_fs_transform, tau, p)
        result_jit = jax.jit(lambda t, x: cxfm.act(circle_fs_transform, t, x))(tau, p)

        np.testing.assert_allclose(
            _as_array(result_jit, "km"), _as_array(result_eager, "km"), atol=1e-10
        )

    def test_act_vmap_over_tau(self, circle_fs_transform) -> None:
        """Act can be vmapped over the tau parameter."""
        taus = u.Q(jnp.linspace(0.0, 2.0, 5), "s")
        p = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")

        results = jax.vmap(lambda t: cxfm.act(circle_fs_transform, t, p))(taus)

        # All results should be 3-vectors
        assert results.shape == (5, 3)


# ===================================================================
# Group F: Active semantics


class TestFrenetSerretActiveSemantics:
    """Active-transform semantics hold for FrenetSerretTransform."""

    def test_forward_moves_point_to_curve_frame(self, circle_fs_transform) -> None:
        """Forward transform moves a point into the curve frame.

        At tau=0, the curve is at (1,0,0) km with T=(0,1,0), N=(-1,0,0).
        The point at the curve origin should map to (0,0,0) in the frame.
        """
        tau = u.Q(0.0, "s")
        p_on_curve = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
        result = cxfm.act(circle_fs_transform, tau, p_on_curve)

        np.testing.assert_allclose(_as_array(result, "km"), [0.0, 0.0, 0.0], atol=1e-6)

    def test_inverse_moves_point_back_to_ambient(self, circle_fs_transform) -> None:
        """Inverse transform moves a curve-frame point back to ambient.

        Origin of curve frame at tau=0 should map back to gamma(0)=(1,0,0).
        """
        tau = u.Q(0.0, "s")
        p_origin = u.Q(jnp.array([0.0, 0.0, 0.0]), "km")
        result = cxfm.act(circle_fs_transform.inverse, tau, p_origin)

        np.testing.assert_allclose(_as_array(result, "km"), [1.0, 0.0, 0.0], atol=1e-6)

    def test_frame_transition_matches_direct_transform(
        self, circle_fs_frame, circle_fs_transform
    ) -> None:
        """frame_transition(Alice, fs_frame) applies the same as the xop."""
        tau = u.Q(0.5, "s")
        p = u.Q(jnp.array([2.0, 1.0, -1.0]), "km")

        via_transition = cxfm.act(
            cxf.frame_transition(cxf.Alice(), circle_fs_frame), tau, p
        )
        via_direct = cxfm.act(circle_fs_transform, tau, p)

        np.testing.assert_allclose(
            _as_array(via_transition, "km"),
            _as_array(via_direct, "km"),
            atol=1e-10,
        )
