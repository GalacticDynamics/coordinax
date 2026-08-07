"""Tests for `interval`, `causal_character`, `proper_time`, `proper_distance`.

These are the Minkowski-correct replacements for the `separation` call that
`norm`'s positive-definiteness guard now refuses. The sharpest test in here is
Lorentz invariance: the interval must be unchanged by a boost, which is the
whole reason it is the right primitive.
"""

from typing import ClassVar

import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinax.representations as cxr
import coordinax.transforms as cxfm

ATOL = 1e-4

ORIGIN = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}


def event(ct, x, y=0.0, z=0.0):
    return {"ct": u.Q(ct, "m"), "x": u.Q(x, "m"), "y": u.Q(y, "m"), "z": u.Q(z, "m")}


class TestInterval:
    """The signed quadratic form, defined where `separation` is not."""

    @pytest.mark.parametrize(
        ("ct", "x", "want"),
        [(5.0, 1.0, -24.0), (1.0, 5.0, 24.0), (3.0, 3.0, 0.0), (0.0, 2.0, 4.0)],
    )
    def test_minkowski_values(self, ct, x, want):
        got = cxm.interval(cxc.minkowskict, ORIGIN, event(ct, x))
        assert float(got.ustrip("m2")) == pytest.approx(want, abs=ATOL)

    def test_timelike_pair_is_finite_not_nan(self):
        """The exact case that made `separation` return ``nan``."""
        got = cxm.interval(cxc.minkowskict, ORIGIN, event(5.0, 1.0))
        assert not bool(jnp.isnan(got.ustrip("m2")))
        assert float(got.ustrip("m2")) < 0

    def test_riemannian_interval_is_squared_separation(self):
        """For a positive-definite metric the two agree, so nothing forks."""
        a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        ds2 = cxm.interval(cxc.cart3d, a, b)
        sep = cxm.separation(cxc.cart3d, a, b)
        assert float(ds2.ustrip("m2")) == pytest.approx(
            float(sep.ustrip("m")) ** 2, abs=ATOL
        )

    def test_symmetric_in_its_arguments(self):
        fwd = cxm.interval(cxc.minkowskict, ORIGIN, event(5.0, 1.0))
        rev = cxm.interval(cxc.minkowskict, event(5.0, 1.0), ORIGIN)
        assert float(fwd.ustrip("m2")) == pytest.approx(
            float(rev.ustrip("m2")), abs=ATOL
        )

    def test_packed_quantity_agrees_with_cdict(self):
        a = u.Q([0.0, 0.0, 0.0, 0.0], "m")
        b = u.Q([5.0, 1.0, 0.0, 0.0], "m")
        packed = cxm.interval(cxc.minkowskict, a, b)
        as_dict = cxm.interval(cxc.minkowskict, ORIGIN, event(5.0, 1.0))
        assert float(packed.ustrip("m2")) == pytest.approx(
            float(as_dict.ustrip("m2")), abs=ATOL
        )

    def test_explicit_metric_overload(self):
        got = cxm.interval(
            cxm.MinkowskiMetric(), cxc.minkowskict, ORIGIN, event(1.0, 5.0)
        )
        assert float(got.ustrip("m2")) == pytest.approx(24.0, abs=ATOL)


class TestLorentzInvariance:
    """The interval is invariant under boosts — its reason for existing."""

    BETAS: ClassVar = [
        [0.6, 0.0, 0.0],
        [0.0, 0.8, 0.0],
        [0.3, -0.4, 0.5],
        [0.99, 0.0, 0.0],
    ]

    @pytest.mark.parametrize("beta", BETAS)
    @pytest.mark.parametrize(
        ("ct", "x"), [(5.0, 1.0), (1.0, 5.0), (3.0, 3.0), (2.0, -7.0)]
    )
    def test_interval_is_unchanged_by_a_boost(self, beta, ct, x):
        op = cxfm.LorentzBoost(beta)
        a, b = ORIGIN, event(ct, x)
        before = cxm.interval(cxc.minkowskict, a, b)

        a2 = cxfm.act(op, None, a, cxc.minkowskict, cxr.point)
        b2 = cxfm.act(op, None, b, cxc.minkowskict, cxr.point)
        after = cxm.interval(cxc.minkowskict, a2, b2)

        assert float(after.ustrip("m2")) == pytest.approx(
            float(before.ustrip("m2")), abs=1e-2
        )

    @pytest.mark.parametrize("beta", BETAS)
    @pytest.mark.parametrize(("ct", "x"), [(5.0, 1.0), (1.0, 5.0), (3.0, 3.0)])
    def test_causal_character_is_boost_invariant(self, beta, ct, x):
        """Causal ordering is absolute: no boost turns timelike into spacelike."""
        op = cxfm.LorentzBoost(beta)
        a, b = ORIGIN, event(ct, x)
        before = cxm.causal_character(cxc.minkowskict, a, b)

        a2 = cxfm.act(op, None, a, cxc.minkowskict, cxr.point)
        b2 = cxfm.act(op, None, b, cxc.minkowskict, cxr.point)
        after = cxm.causal_character(cxc.minkowskict, a2, b2)

        assert after == before

    def test_proper_time_is_boost_invariant(self):
        """A wristwatch reading cannot depend on who is looking at it."""
        a, b = ORIGIN, event(5.0, 1.0)
        before = cxm.proper_time(cxc.minkowskict, a, b).uconvert("s")

        op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        a2 = cxfm.act(op, None, a, cxc.minkowskict, cxr.point)
        b2 = cxfm.act(op, None, b, cxc.minkowskict, cxr.point)
        after = cxm.proper_time(cxc.minkowskict, a2, b2).uconvert("s")

        assert float(after.ustrip("s")) == pytest.approx(
            float(before.ustrip("s")), rel=1e-3
        )


class TestCausalCharacter:
    """Classification by the sign of the interval."""

    @pytest.mark.parametrize(
        ("ct", "x", "want"),
        [
            (5.0, 1.0, "timelike"),
            (1.0, 5.0, "spacelike"),
            (3.0, 3.0, "null"),
            (-4.0, 1.0, "timelike"),
            (0.0, 1.0, "spacelike"),
        ],
    )
    def test_classification(self, ct, x, want):
        assert cxm.causal_character(cxc.minkowskict, ORIGIN, event(ct, x)) == want

    def test_coincident_events_are_null(self):
        assert cxm.causal_character(cxc.minkowskict, ORIGIN, ORIGIN) == "null"

    def test_null_tolerance_scales_with_the_data(self):
        """A light ray a million metres long is still null, not spacelike."""
        big = event(1e6, 1e6)
        assert cxm.causal_character(cxc.minkowskict, ORIGIN, big) == "null"

    def test_explicit_atol_is_respected(self):
        """A wide tolerance can call a genuinely timelike pair null."""
        pair = event(5.0, 1.0)  # ds^2 = -24
        assert cxm.causal_character(cxc.minkowskict, ORIGIN, pair) == "timelike"
        loose = cxm.causal_character(cxc.minkowskict, ORIGIN, pair, atol=100.0)
        assert loose == "null"

    def test_riemannian_metric_is_rejected(self):
        """Causal character is meaningless without a timelike direction."""
        a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        with pytest.raises(ValueError, match="Lorentzian"):
            cxm.causal_character(cxc.cart3d, a, b)


class TestProperTimeAndDistance:
    """Magnitudes, and refusal on the causal type where they are undefined."""

    def test_proper_time_at_rest_is_coordinate_time(self):
        """An observer at rest ages by the full coordinate time."""
        one_light_second = event(299792458.0, 0.0)
        got = cxm.proper_time(cxc.minkowskict, ORIGIN, one_light_second)
        assert float(got.uconvert("s").ustrip("s")) == pytest.approx(1.0, rel=1e-4)

    def test_moving_clock_ages_less(self):
        """Time dilation: same coordinate time, but moving, gives less ageing."""
        c = 299792458.0
        at_rest = cxm.proper_time(cxc.minkowskict, ORIGIN, event(c, 0.0))
        moving = cxm.proper_time(cxc.minkowskict, ORIGIN, event(c, 0.6 * c))
        assert float(moving.ustrip("s")) < float(at_rest.ustrip("s"))
        # sqrt(1 - 0.6^2) = 0.8
        assert float(moving.ustrip("s")) == pytest.approx(
            0.8 * float(at_rest.ustrip("s")), rel=1e-3
        )

    def test_proper_distance_value(self):
        got = cxm.proper_distance(cxc.minkowskict, ORIGIN, event(3.0, 5.0))
        assert float(got.ustrip("m")) == pytest.approx(4.0, abs=ATOL)

    @pytest.mark.parametrize(("ct", "x"), [(1.0, 5.0), (3.0, 3.0)])
    def test_proper_time_refuses_non_timelike(self, ct, x):
        with pytest.raises(ValueError, match="timelike"):
            cxm.proper_time(cxc.minkowskict, ORIGIN, event(ct, x))

    @pytest.mark.parametrize(("ct", "x"), [(5.0, 1.0), (3.0, 3.0)])
    def test_proper_distance_refuses_non_spacelike(self, ct, x):
        with pytest.raises(ValueError, match="spacelike"):
            cxm.proper_distance(cxc.minkowskict, ORIGIN, event(ct, x))

    def test_error_message_names_the_actual_causal_type(self):
        with pytest.raises(ValueError, match="spacelike") as exc:
            cxm.proper_time(cxc.minkowskict, ORIGIN, event(1.0, 5.0))
        assert "spacelike" in str(exc.value)
