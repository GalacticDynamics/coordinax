"""Four curve shapes the arc-length machinery consumes, plus a stitching trap.

Each fixture below is a curve already parametrised by arc length -- the shape
a per-timestep fitter to a stream simulator produces -- rather than a curve
`ArcLength` reparametrises. The four shapes are: a one-argument curve that is
already unit-speed; wrapping an already-unit-speed curve in `ArcLength`
(idempotent, but wasteful); a two-argument, per-slice unit-speed curve read
through `AtTime`; and the station-first, time-second argument convention.

The last section pins a trap: a linear blend of two unit-speed curves is not
itself unit-speed, and shows the remedy -- wrapping the blend in `ArcLength`
re-measures speed per slice.
"""

import astropy.units as apyu
import jax
import jax.numpy as jnp
import pytest
from coordinaxs.api.manifolds import metric_matrix

import coordinax.charts as cxc
import unxt as u

import coordinaxs.curveframes as cxfc

# ---------------------------------------------------------------------------
# Shapes 1 & 2: a one-argument curve that is already arc-length parametrised.


def circle(s: u.AbstractQuantity, radius_km: float = 2.0) -> u.AbstractQuantity:
    """Arc-length circle: gamma(s) = (R cos(s/R), R sin(s/R), 0), unit speed in s."""
    v = s.ustrip("km")
    r = radius_km
    return u.Q(
        jnp.stack([r * jnp.cos(v / r), r * jnp.sin(v / r), jnp.zeros_like(v)]), "km"
    )


def _chart(curve, s_max: float) -> cxfc.TubularChart:
    return cxfc.TubularChart(
        cxfc.BishopBuilder(curve, "km"), tau_bounds=(u.Q(0.0, "km"), u.Q(s_max, "km"))
    )


def test_shape1_already_arclength_curve_has_unit_speed_with_no_wrapper() -> None:
    """A curve that arrives already arc-length parametrised needs no `ArcLength`."""

    def pos(s_val):
        return circle(u.Q(s_val, "km")).ustrip("km")

    speed = jnp.linalg.norm(jax.jacfwd(pos)(1.3))
    assert jnp.allclose(speed, 1.0, atol=1e-10), speed


def test_shape1_g_ss_is_one_through_a_chart() -> None:
    """No speed factor survives into the chart metric: g_ss = 1 on the curve."""
    ch = _chart(circle, 2 * jnp.pi * 2.0)
    at = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
    g = metric_matrix(ch.M, at, ch).matrix
    assert jnp.allclose(g[0, 0].ustrip(""), 1.0, atol=1e-10), g[0, 0]


def test_shape1_round_trip_is_exact() -> None:
    """The chart's forward/inverse maps round-trip an already-arclength curve."""
    ch = _chart(circle, 2 * jnp.pi * 2.0)
    p = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.1, "km"), "n2": u.Q(-0.05, "km")}
    xyz = cxc.pt_map(p, ch.M, ch, ch.M, cxc.cart3d)
    back = cxc.pt_map(xyz, ch.M, cxc.cart3d, ch.M, ch)
    assert jnp.allclose(back["tau"].ustrip("km"), 1.3, atol=1e-6)
    assert jnp.allclose(back["n1"].ustrip("km"), 0.1, atol=1e-6)
    assert jnp.allclose(back["n2"].ustrip("km"), -0.05, atol=1e-6)


def test_shape2_wrapping_an_already_unit_speed_curve_is_idempotent() -> None:
    """`ArcLength` on an already-unit-speed curve reproduces it, near machine epsilon.

    Harmless, but an ODE solve per call the caller did not need -- see Shape 1,
    which gets the same answer for free. This is not something to do
    defensively.
    """
    arc = cxfc.ArcLength(circle, tau_unit="km")
    direct = circle(u.Q(1.3, "km")).ustrip("km")
    wrapped = arc(u.Q(1.3, "km")).ustrip("km")
    diff = float(jnp.max(jnp.abs(direct - wrapped)))
    assert diff < 1e-10, diff


# ---------------------------------------------------------------------------
# Shape 3: a time-series of arc-length curves, unit-speed in s at every t.


def series(s: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """Circle of radius 2+t, arc-length parametrised in s at every slice t."""
    r = 2.0 + t.ustrip("s")
    v = s.ustrip("km")
    return u.Q(
        jnp.stack([r * jnp.cos(v / r), r * jnp.sin(v / r), jnp.zeros_like(v)]), "km"
    )


@pytest.mark.parametrize("t_val", [0.0, 1.0, 3.0])
def test_shape3_time_series_is_unit_speed_at_every_slice(t_val: float) -> None:
    """Each slice of `series` is independently unit-speed in s, at every t."""
    curve_t = cxfc.AtTime(series, u.Q(t_val, "s"))

    def pos(s_val):
        return curve_t(u.Q(s_val, "km")).ustrip("km")

    speed = jnp.linalg.norm(jax.jacfwd(pos)(1.3))
    assert jnp.allclose(speed, 1.0, atol=1e-10), (t_val, speed)


@pytest.mark.parametrize("t_val", [0.0, 1.0, 3.0])
def test_shape3_attime_builder_chart_gives_g_ss_one(t_val: float) -> None:
    """`AtTime(series, t)` feeds a builder and a chart exactly like any curve."""
    curve_t = cxfc.AtTime(series, u.Q(t_val, "s"))
    ch = _chart(curve_t, 30.0)
    at = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
    g = metric_matrix(ch.M, at, ch).matrix
    assert jnp.allclose(g[0, 0].ustrip(""), 1.0, atol=1e-8), (t_val, g[0, 0])


# ---------------------------------------------------------------------------
# Shape 4: argument order is positional -- station first, time second.


def test_shape4_swapped_argument_order_fails() -> None:
    """gamma(t, s) fails because a station is a length and a time is not.

    This is a lucky accident of units catching the mistake, not a designed
    guard -- see the next test.
    """
    with pytest.raises(apyu.UnitConversionError, match="not convertible"):
        series(u.Q(1.0, "s"), u.Q(1.3, "km"))


def two_lengths(s: u.AbstractQuantity, r: u.AbstractQuantity) -> u.AbstractQuantity:
    """A curve whose two parameters, s and r, share a dimension (length)."""
    return u.Q(
        jnp.stack([s.ustrip("km"), r.ustrip("km"), jnp.zeros_like(s.ustrip("km"))]),
        "km",
    )


def test_shape4_same_dimension_params_are_silently_transposed() -> None:
    """When both parameters share a dimension, a swapped call raises nothing.

    It returns a different, silently transposed answer instead -- the failure
    in the previous test is unit-mismatch luck, not something the library
    checks for.
    """
    a, b = u.Q(1.3, "km"), u.Q(4.0, "km")
    forward = two_lengths(a, b).ustrip("km")  # [1.3, 4.0, 0.0]
    swapped = two_lengths(b, a).ustrip("km")  # [4.0, 1.3, 0.0], no error
    assert jnp.allclose(swapped, jnp.array([forward[1], forward[0], forward[2]]))
    assert not jnp.allclose(swapped, forward)


# ---------------------------------------------------------------------------
# The trap: a linear blend of two unit-speed curves is not itself unit-speed.


def blend(s: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """Linear blend, over t in [0, 1], of arc-length circles of radius 2 and 3."""
    v = s.ustrip("km")
    tv = t.ustrip("s")
    c2 = jnp.stack([2.0 * jnp.cos(v / 2.0), 2.0 * jnp.sin(v / 2.0), jnp.zeros_like(v)])
    c3 = jnp.stack([3.0 * jnp.cos(v / 3.0), 3.0 * jnp.sin(v / 3.0), jnp.zeros_like(v)])
    return u.Q((1.0 - tv) * c2 + tv * c3, "km")


_S = 1.3  # the interpolated station at which the drift below was measured
_SAMPLED = (0.0, 1.0)  # timesteps the blend was actually stitched from
_INTERPOLATED = (0.25, 0.5)  # timesteps that only exist by linear interpolation


def _speed_at(t_val: float) -> float:
    curve_t = cxfc.AtTime(blend, u.Q(t_val, "s"))

    def pos(s_val):
        return curve_t(u.Q(s_val, "km")).ustrip("km")

    return float(jnp.linalg.norm(jax.jacfwd(pos)(_S)))


@pytest.mark.parametrize("t_val", _SAMPLED)
def test_trap_sampled_slices_stay_unit_speed(t_val: float) -> None:
    """At a stitched timestep, the blend collapses to one circle: unit speed."""
    assert jnp.allclose(_speed_at(t_val), 1.0, atol=1e-8), t_val


@pytest.mark.parametrize("t_val", _INTERPOLATED)
def test_trap_interpolated_slices_drift_below_unit_speed(t_val: float) -> None:
    """Between the stitched timesteps, the blend is measurably not unit-speed.

    This is the assertion that would catch an accidentally unit-speed
    fixture: if the blend were unit-speed throughout, this would fail instead
    of the remedy test below being meaningful.
    """
    speed = _speed_at(t_val)
    assert speed < 1.0 - 1e-3, (t_val, speed)


def test_trap_propagates_to_a_wrong_g_ss_through_a_chart() -> None:
    """The drift at t=0.5 is not just a speed number -- it reaches the chart metric."""
    curve_half = cxfc.AtTime(blend, u.Q(0.5, "s"))
    ch = _chart(curve_half, 20.0)
    at = {"tau": u.Q(_S, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
    g = metric_matrix(ch.M, at, ch).matrix
    g_ss = float(g[0, 0].ustrip(""))
    assert g_ss < 1.0 - 1e-3, g_ss


@pytest.mark.parametrize("t_val", _INTERPOLATED)
def test_trap_remedy_arclength_wrap_restores_unit_speed(t_val: float) -> None:
    """Wrapping the stitched curve in `ArcLength` re-measures speed per slice."""
    arc = cxfc.ArcLength(blend, tau_unit="km")
    curve_t = cxfc.AtTime(arc, u.Q(t_val, "s"))

    def pos(s_val):
        return curve_t(u.Q(s_val, "km")).ustrip("km")

    speed = jnp.linalg.norm(jax.jacfwd(pos)(_S))
    assert jnp.allclose(speed, 1.0, atol=1e-8), (t_val, speed)


def test_trap_remedy_restores_g_ss_through_a_chart() -> None:
    """The `ArcLength`-wrapped blend gives g_ss = 1 on the same interpolated slice."""
    arc = cxfc.ArcLength(blend, tau_unit="km")
    curve_half = cxfc.AtTime(arc, u.Q(0.5, "s"))
    ch = _chart(curve_half, 20.0)
    at = {"tau": u.Q(_S, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
    g = metric_matrix(ch.M, at, ch).matrix
    assert jnp.allclose(g[0, 0].ustrip(""), 1.0, atol=1e-6), g[0, 0]
