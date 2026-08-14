"""Tests for `interval`, `causal_character`, `proper_time`, `proper_distance`.

These are the Minkowski-correct replacements for the `geodesic_distance` call that
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
import coordinaxs.api.manifolds as cxmapi

ATOL = 1e-4

ORIGIN = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}


def event(ct, x, y=0.0, z=0.0):
    return {"ct": u.Q(ct, "m"), "x": u.Q(x, "m"), "y": u.Q(y, "m"), "z": u.Q(z, "m")}


class TestInterval:
    """The signed quadratic form, defined where `geodesic_distance` is not."""

    @pytest.mark.parametrize(
        ("ct", "x", "want"),
        [(5.0, 1.0, -24.0), (1.0, 5.0, 24.0), (3.0, 3.0, 0.0), (0.0, 2.0, 4.0)],
    )
    def test_minkowski_values(self, ct, x, want):
        got = cxm.interval(cxc.minkowskict, ORIGIN, event(ct, x))
        assert float(got.ustrip("m2")) == pytest.approx(want, abs=ATOL)

    def test_timelike_pair_is_finite_not_nan(self):
        """The exact case that made `geodesic_distance` return ``nan``."""
        got = cxm.interval(cxc.minkowskict, ORIGIN, event(5.0, 1.0))
        assert not bool(jnp.isnan(got.ustrip("m2")))
        assert float(got.ustrip("m2")) < 0

    def test_riemannian_interval_is_squared_separation(self):
        """For a positive-definite metric the two agree, so nothing forks."""
        a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        ds2 = cxm.interval(cxc.cart3d, a, b)
        sep = cxm.geodesic_distance(cxc.cart3d, a, b)
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

    def test_mismatched_metric_is_rejected(self):
        """The `metric` argument is a checked selector, not an ignored one.

        It previously had no effect at all: the matrix always came from
        ``chart.M``, so passing a different metric silently returned the
        chart's answer. Matches `norm`'s metric-level contract.
        """
        with pytest.raises(ValueError, match="needs the chart's own metric"):
            cxm.interval(cxm.FlatMetric(4), cxc.minkowskict, ORIGIN, event(1.0, 5.0))

    def test_bare_arrays_require_usys_like_norm_does(self):
        """Sharing the contraction means sharing its contract.

        `interval` used to answer here while `norm` on the same data correctly
        refused: with no units on the components there is nothing to derive the
        result's unit from, so the caller has to name a unit system. That
        divergence was the reason for routing both through one primitive, so the
        stricter behaviour is the point, not a side effect.
        """
        origin = {k: jnp.asarray(0.0) for k in ("ct", "x", "y", "z")}
        ev = {
            "ct": jnp.asarray(5.0),
            "x": jnp.asarray(1.0),
            "y": jnp.asarray(0.0),
            "z": jnp.asarray(0.0),
        }
        with pytest.raises(TypeError, match=r"interval\(\).*usys"):
            cxm.interval(cxc.minkowskict, origin, ev)

        # ...and with `usys` named, it works and agrees with the Quantity form.
        # Bare arrays in, bare array out -- no units to carry through.
        got = cxm.interval(cxc.minkowskict, origin, ev, usys=u.unitsystems.si)
        assert not hasattr(got, "unit")
        assert float(got) == pytest.approx(-24.0, abs=ATOL)
        assert (
            cxm.lorentzian.causal_character(
                cxc.minkowskict, origin, ev, usys=u.unitsystems.si
            )
            == "timelike"
        )

    def test_error_names_interval_not_the_shared_primitive(self):
        """A caller of `interval` should not be told about `quadratic_form`."""
        origin = {k: jnp.asarray(0.0) for k in ("ct", "x", "y", "z")}
        ev = {k: jnp.asarray(1.0) for k in ("ct", "x", "y", "z")}
        with pytest.raises(TypeError) as exc:
            cxm.interval(cxc.minkowskict, origin, ev)
        assert str(exc.value).startswith("interval()")

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
        before = cxm.lorentzian.causal_character(cxc.minkowskict, a, b)

        a2 = cxfm.act(op, None, a, cxc.minkowskict, cxr.point)
        b2 = cxfm.act(op, None, b, cxc.minkowskict, cxr.point)
        after = cxm.lorentzian.causal_character(cxc.minkowskict, a2, b2)

        assert after == before

    def test_proper_time_is_boost_invariant(self):
        """A wristwatch reading cannot depend on who is looking at it."""
        a, b = ORIGIN, event(5.0, 1.0)
        before = cxm.lorentzian.proper_time(cxc.minkowskict, a, b).uconvert("s")

        op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        a2 = cxfm.act(op, None, a, cxc.minkowskict, cxr.point)
        b2 = cxfm.act(op, None, b, cxc.minkowskict, cxr.point)
        after = cxm.lorentzian.proper_time(cxc.minkowskict, a2, b2).uconvert("s")

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
        assert (
            cxm.lorentzian.causal_character(cxc.minkowskict, ORIGIN, event(ct, x))
            == want
        )

    def test_coincident_events_are_null(self):
        assert (
            cxm.lorentzian.causal_character(cxc.minkowskict, ORIGIN, ORIGIN) == "null"
        )

    def test_null_tolerance_scales_with_the_data(self):
        """A light ray a million metres long is still null, not spacelike."""
        big = event(1e6, 1e6)
        assert cxm.lorentzian.causal_character(cxc.minkowskict, ORIGIN, big) == "null"

    def test_explicit_atol_is_respected(self):
        """A wide tolerance can call a genuinely timelike pair null."""
        pair = event(5.0, 1.0)  # ds^2 = -24
        assert (
            cxm.lorentzian.causal_character(cxc.minkowskict, ORIGIN, pair) == "timelike"
        )
        loose = cxm.lorentzian.causal_character(
            cxc.minkowskict, ORIGIN, pair, atol=100.0
        )
        assert loose == "null"

    @pytest.mark.parametrize(
        "verb", ["causal_character", "proper_time", "proper_distance"]
    )
    def test_non_lorentzian_metric_is_rejected(self, verb):
        """Causal character is meaningless without a timelike direction.

        The refusal is now the *type system's*: there is no method for a
        non-Lorentzian metric, rather than a method that accepts any chart and
        scans ``metric.signature`` at runtime. The `NotImplementedError` below
        comes from a deliberate fallback overload whose only job is to say so in
        a sentence instead of a plum resolution dump.
        """
        a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        with pytest.raises(NotImplementedError, match="requires a Lorentzian metric"):
            getattr(cxm.lorentzian, verb)(cxc.cart3d, a, b)

    def test_the_precondition_is_a_type_not_a_runtime_scan(self):
        """`MinkowskiMetric` carries the marker; Riemannian metrics do not.

        This is what makes the set extensible: a curved spacetime metric
        inherits the marker and acquires all three verbs, rather than each verb
        needing to learn about it.
        """
        assert isinstance(cxm.MinkowskiMetric(), cxm.AbstractLorentzianMetricField)
        assert not isinstance(cxm.FlatMetric(3), cxm.AbstractLorentzianMetricField)
        assert not isinstance(cxm.RoundMetric(2), cxm.AbstractLorentzianMetricField)

    def test_sphere_is_also_rejected(self):
        """Not just flat-Riemannian: any positive-definite metric."""
        at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
        b = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.0, "rad")}
        with pytest.raises(NotImplementedError, match="requires a Lorentzian metric"):
            cxm.lorentzian.causal_character(cxc.sph2, at, b)


class TestProperTimeAndDistance:
    """Magnitudes, and refusal on the causal type where they are undefined."""

    def test_proper_time_at_rest_is_coordinate_time(self):
        """An observer at rest ages by the full coordinate time."""
        one_light_second = event(299792458.0, 0.0)
        got = cxm.lorentzian.proper_time(cxc.minkowskict, ORIGIN, one_light_second)
        assert float(got.uconvert("s").ustrip("s")) == pytest.approx(1.0, rel=1e-4)

    def test_moving_clock_ages_less(self):
        """Time dilation: same coordinate time, but moving, gives less ageing."""
        c = 299792458.0
        at_rest = cxm.lorentzian.proper_time(cxc.minkowskict, ORIGIN, event(c, 0.0))
        moving = cxm.lorentzian.proper_time(cxc.minkowskict, ORIGIN, event(c, 0.6 * c))
        assert float(moving.ustrip("s")) < float(at_rest.ustrip("s"))
        # sqrt(1 - 0.6^2) = 0.8
        assert float(moving.ustrip("s")) == pytest.approx(
            0.8 * float(at_rest.ustrip("s")), rel=1e-3
        )

    def test_proper_distance_value(self):
        got = cxm.lorentzian.proper_distance(cxc.minkowskict, ORIGIN, event(3.0, 5.0))
        assert float(got.ustrip("m")) == pytest.approx(4.0, abs=ATOL)

    @pytest.mark.parametrize(("ct", "x"), [(1.0, 5.0), (3.0, 3.0)])
    def test_proper_time_refuses_non_timelike(self, ct, x):
        with pytest.raises(ValueError, match="timelike"):
            cxm.lorentzian.proper_time(cxc.minkowskict, ORIGIN, event(ct, x))

    @pytest.mark.parametrize(("ct", "x"), [(5.0, 1.0), (3.0, 3.0)])
    def test_proper_distance_refuses_non_spacelike(self, ct, x):
        with pytest.raises(ValueError, match="spacelike"):
            cxm.lorentzian.proper_distance(cxc.minkowskict, ORIGIN, event(ct, x))

    def test_error_message_names_the_actual_causal_type(self):
        with pytest.raises(ValueError, match="spacelike") as exc:
            cxm.lorentzian.proper_time(cxc.minkowskict, ORIGIN, event(1.0, 5.0))
        assert "spacelike" in str(exc.value)


class TestSingleIntervalEvaluation:
    """`proper_time`/`proper_distance` classify from one interval evaluation.

    They used to call `causal_character` (which computes the interval) and then
    `interval` again — two metric-matrix builds for one question. Behaviour must
    be identical after collapsing them to one.
    """

    @pytest.mark.parametrize(("ct", "x"), [(5.0, 1.0), (10.0, 2.0), (-7.0, 3.0)])
    def test_proper_time_agrees_with_the_two_step_form(self, ct, x):
        b = event(ct, x)
        assert cxm.lorentzian.causal_character(cxc.minkowskict, ORIGIN, b) == "timelike"
        ds2 = cxm.interval(cxc.minkowskict, ORIGIN, b)
        expected = float(jnp.sqrt(-ds2.ustrip("m2"))) / 299792458.0
        got = cxm.lorentzian.proper_time(cxc.minkowskict, ORIGIN, b)
        assert float(got.ustrip("s")) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize(("ct", "x"), [(1.0, 5.0), (3.0, 5.0), (0.0, 2.0)])
    def test_proper_distance_agrees_with_the_two_step_form(self, ct, x):
        b = event(ct, x)
        assert (
            cxm.lorentzian.causal_character(cxc.minkowskict, ORIGIN, b) == "spacelike"
        )
        ds2 = cxm.interval(cxc.minkowskict, ORIGIN, b)
        expected = float(jnp.sqrt(ds2.ustrip("m2")))
        got = cxm.lorentzian.proper_distance(cxc.minkowskict, ORIGIN, b)
        assert float(got.ustrip("m")) == pytest.approx(expected, rel=1e-5)

    def test_atol_still_reaches_the_refusal_path(self):
        """The shared classifier still honours `atol` from these entry points."""
        with pytest.raises(ValueError, match="null"):
            cxm.lorentzian.proper_time(
                cxc.minkowskict, ORIGIN, event(5.0, 1.0), atol=100.0
            )


class TestCausalVerbsValidateTheirMetricArgument:
    """The `metric` argument is a checked selector, not an ignored one.

    Regression: the metric-level overloads called the *chart-level* `interval`,
    dropping their own `metric` argument. A Lorentzian metric passed alongside a
    Riemannian chart therefore classified using the chart's metric and slipped
    the precondition entirely:

        causal_character(MinkowskiMetric(), cart3d, a, b)  ->  'spacelike'

    Same defect class as the one review caught in #674's guard and again in
    #680's `interval`; routing through the metric-level `interval` -- which
    validates `metric == chart.M.metric` -- is what fixes it in all three.
    """

    A3: ClassVar = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    B3: ClassVar = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}

    @pytest.mark.parametrize(
        "verb", ["causal_character", "proper_time", "proper_distance"]
    )
    def test_lorentzian_metric_with_riemannian_chart_is_rejected(self, verb):
        """The mismatch must be reported, not silently resolved to the chart."""
        with pytest.raises(ValueError, match="needs the chart's own metric"):
            getattr(cxmapi, verb)(cxm.MinkowskiMetric(), cxc.cart3d, self.A3, self.B3)

    @pytest.mark.parametrize(
        ("verb", "ct", "x"),
        [
            ("causal_character", 5.0, 1.0),  # any pair
            ("proper_time", 5.0, 1.0),  # timelike
            ("proper_distance", 1.0, 5.0),  # spacelike
        ],
    )
    def test_matching_metric_and_chart_still_work(self, verb, ct, x):
        """Positive control: the honest call is unaffected.

        Each verb gets the causal type it is defined for -- `proper_distance`
        refuses a timelike pair by design.
        """
        o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
        got = getattr(cxmapi, verb)(
            cxm.MinkowskiMetric(), cxc.minkowskict, o, event(ct, x)
        )
        assert got is not None
