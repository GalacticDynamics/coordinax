"""Tests for the Bob reference frame."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm


class TestBobExports:
    """Tests that Bob frame and related functions are properly exported."""

    def test_bob_exported(self):
        assert hasattr(cxf, "Bob")
        assert hasattr(cxf, "bob")

    def test_bob_is_instance(self):
        assert isinstance(cxf.bob, cxf.Bob)

    def test_bob_in_all(self):
        assert "Bob" in cxf.__all__
        assert "bob" in cxf.__all__


class TestBobFrameTransitions:
    """Tests for frame transitions involving Bob's frame."""

    def test_bob_to_bob_is_identity(self):
        op = cxf.frame_transition(cxf.bob, cxf.bob)
        assert isinstance(op, cxfm.Identity)

    def test_alice_to_bob_is_composed(self):
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        assert isinstance(op, cxfm.Composed)
        assert len(op.transforms) == 2
        assert isinstance(op.transforms[0], cxfm.Translate)
        assert isinstance(op.transforms[1], cxfm.Boost)

    def test_alice_to_bob_has_correct_chart(self):
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        assert isinstance(op, cxfm.Composed)
        assert op.transforms[0].chart is cxc.cart3d
        assert op.transforms[1].chart is cxc.cart3d

    def test_bob_to_alice_is_composed(self):
        op = cxf.frame_transition(cxf.bob, cxf.alice)
        assert isinstance(op, cxfm.Composed)
        assert len(op.transforms) == 2
        assert isinstance(op.transforms[0], cxfm.Boost)
        assert isinstance(op.transforms[1], cxfm.Translate)

    def test_bob_to_alice_is_inverse_of_alice_to_bob(self):
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        bob_to_alice = cxf.frame_transition(cxf.bob, cxf.alice)
        assert isinstance(alice_to_bob, cxfm.Composed)
        assert isinstance(bob_to_alice, cxfm.Composed)
        assert bob_to_alice == alice_to_bob.inverse

    def test_alice_bob_roundtrip_velocity(self):
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        bob_to_alice = cxf.frame_transition(cxf.bob, cxf.alice)
        v = {"x": u.Q(5.0, "m/s"), "y": u.Q(3.0, "m/s"), "z": u.Q(1.0, "m/s")}
        v_in_bob = cxfm.act(alice_to_bob, None, v, cxc.cart3d, cxr.coord_vel)
        v_back = cast(
            "dict[str, u.AbstractQuantity]",
            cxfm.act(bob_to_alice, None, v_in_bob, cxc.cart3d, cxr.coord_vel),
        )
        for k, val in v.items():
            assert jnp.allclose(
                u.ustrip("m/s", v_back[k]), u.ustrip("m/s", val), atol=1e-6
            )

    def test_alice_to_bob_translates_position(self):
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = cxfm.act(alice_to_bob, None, p, cxc.cart3d, cxr.point)
        expected = {
            "x": u.Q(100001.0, "km"),
            "y": u.Q(10002.0, "km"),
            "z": u.Q(3.0, "km"),
        }
        assert result == expected

    @pytest.mark.parametrize("frames", [("alice",), ("bob",), ("alex",)])
    def test_alice_alice_is_identity(self, frames):
        """Existing frame pair identity transitions still work."""
        frame = getattr(cxf, frames[0])
        op = cxf.frame_transition(frame, frame)
        assert isinstance(op, cxfm.Identity)


# ============================================================================


class TestBobPositionTransform:
    """Positions transform correctly under the Alice ↔ Bob frame transition.

    Per spec (software-spec-bob):
      Point: shifted by [100 000, 10 000, 0] km (Translate component only).
    """

    def test_alice_to_bob_origin(self):
        """Alice's origin maps to Bob's [100 000, 10 000, 0] km."""
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        p = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
        result = cxfm.act(alice_to_bob, None, p, cxc.cart3d, cxr.point)
        assert result == {
            "x": u.Q(100_000.0, "km"),
            "y": u.Q(10_000.0, "km"),
            "z": u.Q(0.0, "km"),
        }

    def test_bob_to_alice_origin(self):
        """Bob's origin maps back to Alice's [-100 000, -10 000, 0] km."""
        bob_to_alice = cxf.frame_transition(cxf.bob, cxf.alice)
        p = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
        result = cxfm.act(bob_to_alice, None, p, cxc.cart3d, cxr.point)
        assert result == {
            "x": u.Q(-100_000.0, "km"),
            "y": u.Q(-10_000.0, "km"),
            "z": u.Q(0.0, "km"),
        }

    def test_alice_bob_roundtrip_position(self):
        """Alice → Bob → Alice is the identity on positions."""
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        bob_to_alice = cxf.frame_transition(cxf.bob, cxf.alice)
        p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        p_in_bob = cxfm.act(alice_to_bob, None, p, cxc.cart3d, cxr.point)
        p_back = cast(
            "dict[str, u.AbstractQuantity]",
            cxfm.act(bob_to_alice, None, p_in_bob, cxc.cart3d, cxr.point),
        )
        for k, val in p.items():
            assert jnp.allclose(
                u.ustrip("km", p_back[k]), u.ustrip("km", val), atol=1e-6
            )

    def test_alice_to_bob_position_jit(self):
        """Alice → Bob position transform is JIT-compatible."""
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)

        @jax.jit
        def transform(p):
            return cxfm.act(alice_to_bob, None, p, cxc.cart3d, cxr.point)

        p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = transform(p)
        assert jnp.allclose(u.ustrip("km", result["x"]), 100_001.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("km", result["y"]), 10_002.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("km", result["z"]), 3.0, atol=1e-6)


class TestBobVelocityTransform:
    """Velocities are boosted by [269 813 212.2, 0, 0] m/s (Alice → Bob).

    Per spec (software-spec-bob):
      Velocity: shifted by [269 813 212.2, 0, 0] m/s (Boost component only).
    """

    _BOOST_X = 269_813_212.2  # m/s

    def test_alice_to_bob_boosts_velocity(self):
        """Velocity x-component is shifted by the boost value."""
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        v = {"x": u.Q(5.0, "m/s"), "y": u.Q(3.0, "m/s"), "z": u.Q(1.0, "m/s")}
        result = cxfm.act(alice_to_bob, None, v, cxc.cart3d, cxr.coord_vel)
        assert jnp.allclose(
            u.ustrip("m/s", result["x"]), 5.0 + self._BOOST_X, rtol=1e-6
        )
        assert jnp.allclose(u.ustrip("m/s", result["y"]), 3.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("m/s", result["z"]), 1.0, atol=1e-6)

    def test_alice_to_bob_velocity_unchanged_by_translate(self):
        """Translate is identity for velocity; only the Boost component acts."""
        # Apply only the Translate step and verify velocity is unchanged.
        shift = cxfm.Translate.from_([100_000, 10_000, 0], "km")
        v = {"x": u.Q(5.0, "m/s"), "y": u.Q(3.0, "m/s"), "z": u.Q(1.0, "m/s")}
        result = cxfm.act(shift, None, v, cxc.cart3d, cxr.coord_vel)
        assert result == v


class TestBobDisplacementInvariance:
    """Displacements are invariant under the Alice ↔ Bob transform.

    Per spec (software-spec-bob):
      Displacement: unchanged (both Translate and Boost are identity on displacements).
    """

    def test_alice_to_bob_displacement_invariant(self):
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        d = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = cxfm.act(alice_to_bob, None, d, cxc.cart3d, cxr.coord_disp)
        assert result == d

    def test_bob_to_alice_displacement_invariant(self):
        bob_to_alice = cxf.frame_transition(cxf.bob, cxf.alice)
        d = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = cxfm.act(bob_to_alice, None, d, cxc.cart3d, cxr.coord_disp)
        assert result == d

    def test_translate_alone_displacement_invariant(self):
        """The Translate component alone is identity on displacements."""
        shift = cxfm.Translate.from_([100_000, 10_000, 0], "km")
        d = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = cxfm.act(shift, None, d, cxc.cart3d, cxr.coord_disp)
        assert result == d

    def test_boost_alone_displacement_invariant(self):
        """The Boost component alone is identity on displacements."""
        boost = cxfm.Boost.from_([269_813_212.2, 0, 0], "m/s")
        d = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = cxfm.act(boost, None, d, cxc.cart3d, cxr.coord_disp)
        assert result == d


class TestBobAccelerationInvariance:
    """Accelerations are invariant under the Alice ↔ Bob transform.

    Per spec (software-spec-bob):
      Acceleration: unchanged (Boost is identity on accelerations).
    """

    def test_alice_to_bob_acceleration_invariant(self):
        alice_to_bob = cxf.frame_transition(cxf.alice, cxf.bob)
        a = {"x": u.Q(1.0, "m/s^2"), "y": u.Q(2.0, "m/s^2"), "z": u.Q(3.0, "m/s^2")}
        result = cxfm.act(alice_to_bob, None, a, cxc.cart3d, cxr.coord_acc)
        assert result == a

    def test_bob_to_alice_acceleration_invariant(self):
        bob_to_alice = cxf.frame_transition(cxf.bob, cxf.alice)
        a = {"x": u.Q(1.0, "m/s^2"), "y": u.Q(2.0, "m/s^2"), "z": u.Q(3.0, "m/s^2")}
        result = cxfm.act(bob_to_alice, None, a, cxc.cart3d, cxr.coord_acc)
        assert result == a
