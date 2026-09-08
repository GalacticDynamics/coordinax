"""Tests for the Bob reference frame.

Bob moves at 0.9 c, so the Alice <-> Bob transition is a Lorentz boost on
spacetime rather than the Galilean velocity kick it used to be. `Carol` is
the slow frame that inherited the kick; see `test_carol_frame.py`.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.manifolds as cxm
import coordinax.transforms as cxfm

C_M_S = 299792458.0


def event(ct, x, y=0.0, z=0.0):
    return {"ct": u.Q(ct, "m"), "x": u.Q(x, "m"), "y": u.Q(y, "m"), "z": u.Q(z, "m")}


class TestBobExports:
    """Bob and its singleton are exported."""

    def test_bob_exported(self):
        assert hasattr(cxf, "Bob")
        assert hasattr(cxf, "bob")

    def test_bob_in_all(self):
        assert "Bob" in cxf.__all__
        assert "bob" in cxf.__all__


class TestBobIsALorentzBoost:
    """The transition is a boost, not a velocity kick."""

    def test_bob_to_bob_is_identity(self):
        assert isinstance(cxf.frame_transition(cxf.bob, cxf.bob), cxfm.Identity)

    def test_alice_to_bob_is_translate_then_boost(self):
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        assert isinstance(op, cxfm.Composed)
        assert isinstance(op.transforms[0], cxfm.Translate)
        assert isinstance(op.transforms[1], cxfm.LorentzBoost)

    def test_it_acts_on_spacetime_not_a_spatial_chart(self):
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        assert op.transforms[0].chart is cxc.minkowskict

    def test_the_boost_carries_bobs_speed(self):
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        assert jnp.allclose(
            jnp.asarray(op.transforms[1].beta), jnp.asarray([0.9, 0.0, 0.0])
        )

    def test_bob_to_alice_is_the_inverse(self):
        there = cxf.frame_transition(cxf.alice, cxf.bob)
        back = cxf.frame_transition(cxf.bob, cxf.alice)
        assert back == there.inverse


class TestBobPreservesTheInterval:
    """The reason a boost is the right operator: it leaves ds^2 alone.

    A Galilean velocity kick does not, which is the physical content of the
    change -- not merely that 0.9 c is a large number.
    """

    PAIRS: ClassVar = [(5.0, 1.0), (1.0, 5.0), (3.0, 3.0), (2.0, -4.0)]

    @pytest.mark.parametrize(("ct", "x"), PAIRS)
    def test_the_boost_leaves_the_interval_alone(self, ct, x):
        """Sharp form: the boost alone, to 1e-12."""
        boost = cxf.frame_transition(cxf.alice, cxf.bob).transforms[1]
        o, e = event(0.0, 0.0), event(ct, x)
        before = cxm.interval(cxc.minkowskict, o, e)
        after = cxm.interval(
            cxc.minkowskict,
            cxfm.act(boost, None, o, cxc.minkowskict, cx.point),
            cxfm.act(boost, None, e, cxc.minkowskict, cx.point),
        )
        assert jnp.allclose(u.ustrip("m2", before), u.ustrip("m2", after), rtol=1e-12)

    @pytest.mark.parametrize(("ct", "x"), PAIRS)
    def test_the_whole_transition_does_too(self, ct, x):
        """Looser, and the looseness is arithmetic rather than physics.

        The translation carries events out to ~1e8 m, so the interval becomes a
        difference of squares of numbers that size: float64 has ~1e-16 relative
        precision, which is ~1e-7 absolute once squared and differenced. That
        cancellation, not the transform, sets the tolerance here -- the sharp
        assertion above is the one about the physics.
        """
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        o, e = event(0.0, 0.0), event(ct, x)
        before = cxm.interval(cxc.minkowskict, o, e)
        after = cxm.interval(
            cxc.minkowskict,
            cxfm.act(op, None, o, cxc.minkowskict, cx.point),
            cxfm.act(op, None, e, cxc.minkowskict, cx.point),
        )
        assert jnp.allclose(u.ustrip("m2", before), u.ustrip("m2", after), atol=1e-5)

    def test_a_null_separation_stays_null(self):
        """Light stays light in every frame -- the sharpest form of the claim."""
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        o, e = event(0.0, 0.0), event(3.0, 3.0)
        after = cxm.interval(
            cxc.minkowskict,
            cxfm.act(op, None, o, cxc.minkowskict, cx.point),
            cxfm.act(op, None, e, cxc.minkowskict, cx.point),
        )
        assert jnp.allclose(u.ustrip("m2", after), 0.0, atol=1e-6)


class TestWhyNotAVelocityKick:
    """Pins the arithmetic that makes the Galilean treatment wrong here."""

    def test_adding_velocities_at_bobs_speed_exceeds_c(self):
        galilean = 0.3 * C_M_S + 0.9 * C_M_S
        assert galilean > C_M_S
        assert jnp.allclose(galilean / C_M_S, 1.2, rtol=1e-9)

    def test_composing_them_does_not(self):
        relativistic = (0.3 + 0.9) / (1 + 0.3 * 0.9)
        assert relativistic < 1.0
        assert jnp.allclose(relativistic, 0.9449, atol=1e-4)


class TestBobRefusesPurelySpatialInput:
    """A boost needs a time component; a 3-D point has none."""

    def test_a_3d_point_has_no_time_to_boost(self):
        """Named, not a broad catch: the refusal *is* the contract here.

        Anything narrower than the point's manifold failing to match the
        transition's would mean something else went wrong.
        """
        op = cxf.frame_transition(cxf.alice, cxf.bob)
        with pytest.raises(cxc.ManifoldMismatchError, match="no transition"):
            op(cx.Point.from_([0.0, 0.0, 0.0], "m"))


class TestBobRoundTrips:
    """Alice -> Bob -> Alice returns the event it started from."""

    def test_alice_bob_alice_returns_the_event(self):
        there = cxf.frame_transition(cxf.alice, cxf.bob)
        back = cxf.frame_transition(cxf.bob, cxf.alice)
        e = event(2.0, 1.0, 0.5, -1.0)
        moved = cxfm.act(there, None, e, cxc.minkowskict, cx.point)
        home = cxfm.act(back, None, moved, cxc.minkowskict, cx.point)
        for k in ("ct", "x", "y", "z"):
            assert jnp.allclose(u.ustrip("m", home[k]), u.ustrip("m", e[k]), atol=1e-6)

    def test_the_transition_is_jit_compatible(self):
        op = cxf.frame_transition(cxf.alice, cxf.bob)

        @jax.jit
        def go(e):
            return cxfm.act(op, None, e, cxc.minkowskict, cx.point)

        out = go(event(2.0, 1.0))
        assert jnp.isfinite(u.ustrip("m", out["ct"]))
