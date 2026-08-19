"""Bring-your-own curve types through the arc-length machinery.

``test_arclength_shapes.py`` covers the four parameterisation shapes built from
plain functions. This module is the complement: the same shapes, but built
from a user's own class -- an `equinox.Module` -- rather than a plain
function. A curve is consumed purely as a callable throughout
`coordinaxs.curveframes`, so any callable works, including one backed by
sampled data (e.g. knots and positions interpolated with `jnp.interp`).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from coordinaxs.api.manifolds import metric_matrix

import unxt as u

import coordinaxs.curveframes as cxfc

_AT = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}


def _g_ss(curve, tau_unit: str, s_max: float) -> float:
    chart = cxfc.TubularChart(
        cxfc.BishopBuilder(curve, tau_unit),
        tau_bounds=(u.Q(0.0, tau_unit), u.Q(s_max, tau_unit)),
    )
    g = metric_matrix(chart.M, _AT, chart).matrix
    return float(g[0, 0].ustrip(""))


# ---------------------------------------------------------------------------
# BYO curve in time: an eqx.Module parametrised by time.


class TimeCircle(eqx.Module):
    """A circle of a given radius, parametrised by time."""

    radius: u.AbstractQuantity

    def __call__(self, tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        r = self.radius.ustrip("km")
        return u.Q(jnp.stack([r * jnp.cos(t), r * jnp.sin(t), jnp.zeros_like(t)]), "km")


def test_byo_time_curve_is_not_unit_speed() -> None:
    """A time-parametrised BYO curve is not unit-speed: speed equals the radius."""
    curve = TimeCircle(radius=u.Q(2.0, "km"))

    def pos(t_val):
        return curve(u.Q(t_val, "s")).ustrip("km")

    speed = float(jnp.linalg.norm(jax.jacfwd(pos)(0.7)))
    assert jnp.allclose(speed, 2.0, atol=1e-10), speed


def test_byo_time_curve_wrapped_gives_unit_speed_metric() -> None:
    """Wrapping the BYO time curve in `ArcLength(curve, "s")` gives g_ss = 1."""
    curve = TimeCircle(radius=u.Q(2.0, "km"))
    arc = cxfc.ArcLength(curve, "s")  # the curve's own parameter is a time
    g_ss = _g_ss(arc, "km", 10.0)
    assert jnp.allclose(g_ss, 1.0, atol=1e-8), g_ss


# ---------------------------------------------------------------------------
# BYO curve in arc length: an eqx.Module already parametrised by length.


class ArcCircle(eqx.Module):
    """A circle of a given radius, already parametrised by arc length."""

    radius: u.AbstractQuantity

    def __call__(self, s: u.AbstractQuantity) -> u.AbstractQuantity:
        v = s.ustrip("km")
        r = self.radius.ustrip("km")
        return u.Q(
            jnp.stack([r * jnp.cos(v / r), r * jnp.sin(v / r), jnp.zeros_like(v)]), "km"
        )


def test_byo_arclength_curve_is_unit_speed_with_no_wrapper() -> None:
    """A BYO curve already parametrised by arc length is unit-speed as-is."""
    curve = ArcCircle(radius=u.Q(2.0, "km"))

    def pos(s_val):
        return curve(u.Q(s_val, "km")).ustrip("km")

    speed = float(jnp.linalg.norm(jax.jacfwd(pos)(1.3)))
    assert jnp.allclose(speed, 1.0, atol=1e-10), speed


def test_byo_arclength_curve_metric_is_one_with_no_wrapper() -> None:
    """With no wrapper, the BYO arc-length curve gives g_ss = 1."""
    curve = ArcCircle(radius=u.Q(2.0, "km"))
    g_ss = _g_ss(curve, "km", 10.0)
    assert jnp.allclose(g_ss, 1.0, atol=1e-10), g_ss


def test_byo_arclength_curve_requires_tau_unit() -> None:
    """`tau_unit` has no default, so a length-parametrised curve cannot omit it.

    `ArcLength(my_arc_curve)` with no `tau_unit` fails at construction --
    before any ODE solve gets a chance to convert the length-valued `s` into
    the wrong unit deep inside `__call__`.
    """
    curve = ArcCircle(radius=u.Q(2.0, "km"))
    with pytest.raises(TypeError, match="tau_unit"):
        cxfc.ArcLength(curve)


def test_byo_arclength_curve_explicit_tau_unit_works() -> None:
    """`ArcLength(my_arc_curve, "km")` -- the correct call -- reproduces the curve."""
    curve = ArcCircle(radius=u.Q(2.0, "km"))
    arc = cxfc.ArcLength(curve, "km")
    direct = curve(u.Q(1.3, "km")).ustrip("km")
    wrapped = arc(u.Q(1.3, "km")).ustrip("km")
    assert jnp.allclose(direct, wrapped, atol=1e-10)


# ---------------------------------------------------------------------------
# BYO combo: a two-argument eqx.Module gamma(s, t), unit-speed in s per slice.


class SeriesOne(eqx.Module):
    """A one-argument BYO curve, for arity-detection contrast with `SeriesTwo`."""

    def __call__(self, s: u.AbstractQuantity) -> u.AbstractQuantity:
        v = s.ustrip("km")
        return u.Q(jnp.stack([v, jnp.zeros_like(v), jnp.zeros_like(v)]), "km")


class SeriesTwo(eqx.Module):
    """A two-argument BYO curve gamma(s, t): a circle of radius 2+t, unit-speed in s."""

    def __call__(
        self, s: u.AbstractQuantity, t: u.AbstractQuantity
    ) -> u.AbstractQuantity:
        r = 2.0 + t.ustrip("s")
        v = s.ustrip("km")
        return u.Q(
            jnp.stack([r * jnp.cos(v / r), r * jnp.sin(v / r), jnp.zeros_like(v)]), "km"
        )


def test_byo_arity_detection_matches_signature() -> None:
    """`ArcLength` detects one- vs two-argument BYO classes via `inspect.signature`.

    `inspect.signature` on a bound `__call__` drops `self`, so a one-argument
    class reports `_two_argument=False` and a two-argument one reports
    `_two_argument=True`, exactly as for plain functions.
    """
    one_arg = cxfc.ArcLength(SeriesOne(), "km")
    two_arg = cxfc.ArcLength(SeriesTwo(), "km")
    assert one_arg._two_argument is False
    assert two_arg._two_argument is True


def test_byo_combo_curve_is_unit_speed_via_attime() -> None:
    """The BYO combo curve, bound to a slice with `AtTime`, is unit-speed in s."""
    curve_t = cxfc.AtTime(SeriesTwo(), u.Q(1.0, "s"))

    def pos(s_val):
        return curve_t(u.Q(s_val, "km")).ustrip("km")

    speed = float(jnp.linalg.norm(jax.jacfwd(pos)(1.3)))
    assert jnp.allclose(speed, 1.0, atol=1e-10), speed


def test_byo_combo_curve_metric_is_one_via_attime() -> None:
    """The BYO combo curve, bound with `AtTime` and charted, gives g_ss = 1."""
    curve_t = cxfc.AtTime(SeriesTwo(), u.Q(1.0, "s"))
    g_ss = _g_ss(curve_t, "km", 10.0)
    assert jnp.allclose(g_ss, 1.0, atol=1e-8), g_ss


# ---------------------------------------------------------------------------
# BYO backed by sampled data: knots and positions interpolated with jnp.interp.
#
# Uniform sampling of a circle of radius 2 km over 400 knots. A chord between
# two knots is always shorter than the arc it approximates, so the
# piecewise-linear interpolant is measurably slower than unit speed between
# knots -- the realistic shape a fitter's sampled output takes.

_SAMPLE_RADIUS = 2.0
_SAMPLE_N = 400
_SAMPLE_S = 1.3


class SampledCurve(eqx.Module):
    """A curve reconstructed from arc-length knots and positions via `jnp.interp`."""

    knots: jax.Array
    xs: jax.Array
    ys: jax.Array
    zs: jax.Array

    def __call__(self, s: u.AbstractQuantity) -> u.AbstractQuantity:
        v = s.ustrip("km")
        x = jnp.interp(v, self.knots, self.xs)
        y = jnp.interp(v, self.knots, self.ys)
        z = jnp.interp(v, self.knots, self.zs)
        return u.Q(jnp.stack([x, y, z]), "km")


def _sampled_circle() -> SampledCurve:
    theta = jnp.linspace(0.0, 2 * jnp.pi, _SAMPLE_N + 1)
    knots = _SAMPLE_RADIUS * theta
    return SampledCurve(
        knots=knots,
        xs=_SAMPLE_RADIUS * jnp.cos(theta),
        ys=_SAMPLE_RADIUS * jnp.sin(theta),
        zs=jnp.zeros_like(theta),
    )


def _speed(curve, s_val: float) -> float:
    def pos(x):
        return curve(u.Q(x, "km")).ustrip("km")

    return float(jnp.linalg.norm(jax.jacfwd(pos)(s_val)))


def test_byo_sampled_data_reference_curve_is_exactly_unit_speed() -> None:
    """The exact analytic circle at the same radius and station is unit-speed.

    This is the known-exact fixture the sampled-data tests below are checked
    against: if the sampled curve's speed came out indistinguishable from
    this one, the chord-error tests would be proving nothing.
    """
    exact = ArcCircle(radius=u.Q(_SAMPLE_RADIUS, "km"))
    assert jnp.allclose(_speed(exact, _SAMPLE_S), 1.0, atol=1e-10)


def test_byo_sampled_data_chords_fall_short_of_unit_speed() -> None:
    """Piecewise-linear chords through the sampled circle fall short of unit speed."""
    speed = _speed(_sampled_circle(), _SAMPLE_S)
    assert speed == pytest.approx(0.999990, abs=5e-7)
    assert speed < 1.0 - 1e-5, speed  # the error is real, not solver noise


def test_byo_sampled_data_metric_reflects_chord_error() -> None:
    """With no wrapper, the sampled curve's g_ss falls short of 1 by the same error."""
    g_ss = _g_ss(_sampled_circle(), "km", float(_SAMPLE_RADIUS * 2 * jnp.pi))
    assert g_ss == pytest.approx(0.999979, abs=5e-7)
    assert g_ss < 1.0 - 1e-5, g_ss


def test_byo_sampled_data_wrap_restores_unit_speed() -> None:
    """Wrapping the sampled curve in `ArcLength` restores unit speed."""
    arc = cxfc.ArcLength(_sampled_circle(), "km")
    assert jnp.allclose(_speed(arc, _SAMPLE_S), 1.0, atol=1e-8)


def test_byo_sampled_data_wrap_restores_metric() -> None:
    """Wrapping the sampled curve in `ArcLength` restores g_ss = 1 through the chart."""
    arc = cxfc.ArcLength(_sampled_circle(), "km")
    g_ss = _g_ss(arc, "km", float(_SAMPLE_RADIUS * 2 * jnp.pi))
    assert jnp.allclose(g_ss, 1.0, atol=1e-6), g_ss


# ---------------------------------------------------------------------------
# Differentiability through a user's fitted parameter.


def test_byo_gradient_through_arclength_matches_finite_difference() -> None:
    """`jax.grad` reaches an `eqx.Module` field through `ArcLength`'s ODE solve."""

    def x_of_radius(radius_km):
        curve = TimeCircle(radius=u.Q(radius_km, "km"))
        arc = cxfc.ArcLength(curve, "s")
        return arc(u.Q(_SAMPLE_S, "km")).ustrip("km")[0]

    analytic = float(jax.grad(x_of_radius)(2.0))
    h = 1e-4
    numeric = (x_of_radius(2.0 + h) - x_of_radius(2.0 - h)) / (2 * h)
    assert analytic == pytest.approx(1.189454962, abs=5e-9)
    assert jnp.allclose(analytic, numeric, atol=1e-6), (analytic, numeric)
