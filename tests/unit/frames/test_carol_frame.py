"""Tests for the Carol reference frame."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm


class TestCarolExports:
    """Tests that Carol frame and related functions are properly exported."""

    def test_carol_exported(self):
        assert hasattr(cxf, "Carol")
        assert hasattr(cxf, "carol")

    def test_carol_is_instance(self):
        assert isinstance(cxf.carol, cxf.Carol)

    def test_carol_in_all(self):
        assert "Carol" in cxf.__all__
        assert "carol" in cxf.__all__


class TestCarolFrameTransitions:
    """Tests for frame transitions involving Carol's frame."""

    def test_carol_to_carol_is_identity(self):
        op = cxf.frame_transition(cxf.carol, cxf.carol)
        assert isinstance(op, cxfm.Identity)

    def test_alice_to_carol_is_composed(self):
        op = cxf.frame_transition(cxf.alice, cxf.carol)
        assert isinstance(op, cxfm.Composed)
        assert len(op.transforms) == 2
        assert isinstance(op.transforms[0], cxfm.Translate)
        # The velocity offset is a fibre-only kick: Translate(semantic_kind=vel).
        assert isinstance(op.transforms[1], cxfm.Translate)
        assert op.transforms[1].semantic_kind == cxr.vel

    def test_alice_to_carol_has_correct_chart(self):
        op = cxf.frame_transition(cxf.alice, cxf.carol)
        assert isinstance(op, cxfm.Composed)
        assert op.transforms[0].chart is cxc.cart3d
        assert op.transforms[1].chart is cxc.cart3d

    def test_carol_to_alice_is_composed(self):
        op = cxf.frame_transition(cxf.carol, cxf.alice)
        assert isinstance(op, cxfm.Composed)
        assert len(op.transforms) == 2
        assert isinstance(op.transforms[0], cxfm.Translate)
        assert op.transforms[0].semantic_kind == cxr.vel
        assert isinstance(op.transforms[1], cxfm.Translate)

    def test_carol_to_alice_is_inverse_of_alice_to_carol(self):
        alice_to_carol = cxf.frame_transition(cxf.alice, cxf.carol)
        carol_to_alice = cxf.frame_transition(cxf.carol, cxf.alice)
        assert isinstance(alice_to_carol, cxfm.Composed)
        assert isinstance(carol_to_alice, cxfm.Composed)
        assert carol_to_alice == alice_to_carol.inverse

    def test_alice_carol_roundtrip_velocity(self):
        alice_to_carol = cxf.frame_transition(cxf.alice, cxf.carol)
        carol_to_alice = cxf.frame_transition(cxf.carol, cxf.alice)
        v = {"x": u.Q(5.0, "m/s"), "y": u.Q(3.0, "m/s"), "z": u.Q(1.0, "m/s")}
        v_in_carol = cxfm.act(alice_to_carol, None, v, cxc.cart3d, cxr.coord_vel)
        v_back = cast(
            "dict[str, u.AbstractQuantity]",
            cxfm.act(carol_to_alice, None, v_in_carol, cxc.cart3d, cxr.coord_vel),
        )
        for k, val in v.items():
            assert jnp.allclose(
                u.ustrip("m/s", v_back[k]), u.ustrip("m/s", val), atol=1e-6
            )

    def test_alice_to_carol_translates_position(self):
        alice_to_carol = cxf.frame_transition(cxf.alice, cxf.carol)
        p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = cxfm.act(alice_to_carol, None, p, cxc.cart3d, cxr.point)
        expected = {
            "x": u.Q(100001.0, "km"),
            "y": u.Q(10002.0, "km"),
            "z": u.Q(3.0, "km"),
        }
        assert result == expected

    @pytest.mark.parametrize("frames", [("alice",), ("carol",), ("alex",)])
    def test_alice_alice_is_identity(self, frames):
        """Existing frame pair identity transitions still work."""
        frame = getattr(cxf, frames[0])
        op = cxf.frame_transition(frame, frame)
        assert isinstance(op, cxfm.Identity)


# ============================================================================


class TestCarolPositionTransform:
    """Positions transform correctly under the Alice ↔ Carol frame transition.

    Per spec (software-spec-carol):
      Point: shifted by [100 000, 10 000, 0] km (Translate component only).
    """

    def test_alice_to_carol_origin(self):
        """Alice's origin maps to Carol's [100 000, 10 000, 0] km."""
        alice_to_carol = cxf.frame_transition(cxf.alice, cxf.carol)
        p = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
        result = cxfm.act(alice_to_carol, None, p, cxc.cart3d, cxr.point)
        assert result == {
            "x": u.Q(100_000.0, "km"),
            "y": u.Q(10_000.0, "km"),
            "z": u.Q(0.0, "km"),
        }

    def test_carol_to_alice_origin(self):
        """Carol's origin maps back to Alice's [-100 000, -10 000, 0] km."""
        carol_to_alice = cxf.frame_transition(cxf.carol, cxf.alice)
        p = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
        result = cxfm.act(carol_to_alice, None, p, cxc.cart3d, cxr.point)
        assert result == {
            "x": u.Q(-100_000.0, "km"),
            "y": u.Q(-10_000.0, "km"),
            "z": u.Q(0.0, "km"),
        }

    def test_alice_carol_roundtrip_position(self):
        """Alice → Carol → Alice is the identity on positions."""
        alice_to_carol = cxf.frame_transition(cxf.alice, cxf.carol)
        carol_to_alice = cxf.frame_transition(cxf.carol, cxf.alice)
        p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        p_in_carol = cxfm.act(alice_to_carol, None, p, cxc.cart3d, cxr.point)
        p_back = cast(
            "dict[str, u.AbstractQuantity]",
            cxfm.act(carol_to_alice, None, p_in_carol, cxc.cart3d, cxr.point),
        )
        for k, val in p.items():
            assert jnp.allclose(
                u.ustrip("km", p_back[k]), u.ustrip("km", val), atol=1e-6
            )

    def test_alice_to_carol_position_jit(self):
        """Alice → Carol position transform is JIT-compatible."""
        alice_to_carol = cxf.frame_transition(cxf.alice, cxf.carol)

        @jax.jit
        def transform(p):
            return cxfm.act(alice_to_carol, None, p, cxc.cart3d, cxr.point)

        p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
        result = transform(p)
        assert jnp.allclose(u.ustrip("km", result["x"]), 100_001.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("km", result["y"]), 10_002.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("km", result["z"]), 3.0, atol=1e-6)


class TestCarolVelocityTransform:
    """Velocities are kicked by [30, 0, 0] km/s (Alice → Carol).

    Per spec (software-spec-carol):
      Velocity: shifted by [30, 0, 0] km/s (kick component only).
    """

    _KICK_X = 30_000.0  # m/s

    def test_alice_to_carol_kicks_velocity(self):
        """Velocity x-component is shifted by the kick value."""
        alice_to_carol = cxf.frame_transition(cxf.alice, cxf.carol)
        v = {"x": u.Q(5.0, "m/s"), "y": u.Q(3.0, "m/s"), "z": u.Q(1.0, "m/s")}
        result = cxfm.act(alice_to_carol, None, v, cxc.cart3d, cxr.coord_vel)
        assert jnp.allclose(u.ustrip("m/s", result["x"]), 5.0 + self._KICK_X, rtol=1e-6)
        assert jnp.allclose(u.ustrip("m/s", result["y"]), 3.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("m/s", result["z"]), 1.0, atol=1e-6)

    def test_alice_to_carol_velocity_unchanged_by_translate(self):
        """Translate is identity for velocity; only the kick acts."""
        # Apply only the Translate step and verify velocity is unchanged.
        shift = cxfm.Translate.from_([100_000, 10_000, 0], "km")
        v = {"x": u.Q(5.0, "m/s"), "y": u.Q(3.0, "m/s"), "z": u.Q(1.0, "m/s")}
        result = cxfm.act(shift, None, v, cxc.cart3d, cxr.coord_vel)
        assert result == v


class TestCarolInvariance:
    """Displacements and accelerations are unchanged by the Alice <-> Carol maps.

    Per spec (software-spec-carol):
      Displacement: unchanged (both offsets are identity on displacements).
      Acceleration: unchanged (a constant velocity kick is identity on
      accelerations).

    The two spec lines are one contract -- ``act`` is the identity on these
    representations -- so they are one table.

    Note what the parametrization covers. Alice <-> Carol is
    ``Translate | Translate(semantic_kind=vel)``: a spatial offset and a
    *velocity kick*, not a `~coordinax.transforms.Boost`. The standalone
    `Boost` case is here as well because the same invariance is claimed of it,
    and testing it alone is the only place that claim actually bites -- but it
    is an addition to the transition's own components, not one of them.
    """

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(
                cxf.frame_transition(cxf.alice, cxf.carol), id="alice-to-carol"
            ),
            pytest.param(
                cxf.frame_transition(cxf.carol, cxf.alice), id="carol-to-alice"
            ),
            pytest.param(
                cxfm.Translate.from_([100_000, 10_000, 0], "km"), id="translate-alone"
            ),
            pytest.param(cxfm.Boost.from_([30_000.0, 0, 0], "m/s"), id="boost-alone"),
        ],
    )
    @pytest.mark.parametrize(
        ("rep", "value"),
        [
            pytest.param(
                cxr.coord_disp,
                {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")},
                id="displacement",
            ),
            pytest.param(
                cxr.coord_acc,
                {
                    "x": u.Q(1.0, "m/s^2"),
                    "y": u.Q(2.0, "m/s^2"),
                    "z": u.Q(3.0, "m/s^2"),
                },
                id="acceleration",
            ),
        ],
    )
    def test_act_is_the_identity(self, transform, rep, value) -> None:
        assert cxfm.act(transform, None, value, cxc.cart3d, rep) == value
