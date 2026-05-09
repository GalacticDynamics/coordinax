"""Tests for the Bob reference frame."""

from typing import cast

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
